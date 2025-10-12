import os
import sys
import gc
import math
import time
import random
import types
import logging
import traceback
from contextlib import contextmanager
from functools import partial

from PIL import Image
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import cv2
import numpy as np

from wan.text2video import (WanT2V, T5EncoderModel, WanVAE, shard_model, FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps, FlowUniPCMultistepScheduler)
from .modules.model import VaceWanModel
from ..utils.preprocessor import VaceVideoProcessor


class WanVaceDual(WanT2V):
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        """
        Enhanced WanVace with dual condition processing capability.
        Process two conditions separately through vace_blocks, then fuse the hints.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating VaceWanModel from {checkpoint_dir}")
        self.model = VaceWanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward,
                                                            usp_dit_forward_vace)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            for block in self.model.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.model.forward_vace = types.MethodType(usp_dit_forward_vace, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

        self.vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(config.vae_stride, self.patch_size)]),
            min_area=480 * 832,
            max_area=480 * 832,
            min_fps=self.config.sample_fps,
            max_fps=self.config.sample_fps,
            zero_start=True,
            seq_len=32760,
            keep_last=True)

    
    def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = vae.encode(frames)
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = vae.encode(inactive)
            reactive = vae.encode(reactive)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs)
                else:
                    ref_latent = vae.encode(refs)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None, vae_stride=None):
        vae_stride = self.vae_stride if vae_stride is None else vae_stride
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // vae_stride[0])
            height = 2 * (int(height) // (vae_stride[1] * 2))
            width = 2 * (int(width) // (vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, vae_stride[1], width, vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                vae_stride[1] * vae_stride[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
       
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def prepare_source(self, src_video, src_mask, src_ref_images, num_frames, image_size, device):
        
        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720*1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480*832:
            self.vid_proc.set_seq_len(32760)
        else:
            raise NotImplementedError(f'image_size {image_size} is not supported')

        image_size = (image_size[1], image_size[0])
        image_sizes = []
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video, sub_src_mask)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(src_video[i].shape[2:])

        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        ref_img = Image.open(ref_img).convert("RGB")
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            ref_height, ref_width = ref_img.shape[-2:]
                            white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device)
                            scale = min(canvas_height / ref_height, canvas_width / ref_width)
                            new_height = int(ref_height * scale)
                            new_width = int(ref_width * scale)
                            resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                            top = (canvas_height - new_height) // 2
                            left = (canvas_width - new_width) // 2
                            white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
                            ref_img = white_canvas
                        src_ref_images[i][j] = ref_img.to(device)
        return src_video, src_mask, src_ref_images

    def decode_latent(self, zs, ref_images=None, vae=None):
       
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(zs)
        else:
            assert len(zs) == len(ref_images)

        trimed_zs = []
        for z, refs in zip(zs, ref_images):
            if refs is not None:
                z = z[:, len(refs):, :, :]
            trimed_zs.append(z)

        return vae.decode(trimed_zs)

    def process_mask_dilate_blur(self, mask_tensor, dilate_pixels=10, blur_sigma=4.0):
        
        if dilate_pixels == 0 and blur_sigma <= 0.01:
            logging.info("Skipping mask processing (dilate_pixels=0, blur_sigma≈0)")
            return mask_tensor
        
        mask_np = mask_tensor.squeeze(0).cpu().numpy()  # (T, H, W)
        processed_frames = []
        
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_pixels*2+1, dilate_pixels*2+1))
        
        for t in range(mask_np.shape[0]):
            frame = mask_np[t]  # (H, W)
            
            
            frame_uint8 = (frame * 255).astype(np.uint8)
            
            
            dilated = cv2.dilate(frame_uint8, kernel, iterations=1)
            
            
            kernel_size = int(blur_sigma * 6 + 1)  
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(dilated, (kernel_size, kernel_size), blur_sigma)
            
            
            processed_frame = blurred.astype(np.float32) / 255.0
            processed_frames.append(processed_frame)
        
       
        processed_mask = torch.tensor(np.stack(processed_frames), dtype=torch.float32, device=mask_tensor.device)
        processed_mask = processed_mask.unsqueeze(0)  # 恢复 (C, T, H, W)
        
        return processed_mask

    def _temporal_interpolation(self, tensor, target_frames):
        
        C, T, H, W = tensor.shape
        time_indices = torch.linspace(0, T - 1, target_frames, device=tensor.device)
        time_indices_floor = time_indices.floor().long()
        time_indices_ceil = time_indices.ceil().long()
        time_weights = time_indices - time_indices_floor.float()
        
        time_indices_floor = torch.clamp(time_indices_floor, 0, T - 1)
        time_indices_ceil = torch.clamp(time_indices_ceil, 0, T - 1)
        
        tensor_new = []
        for t_new in range(target_frames):
            t_floor = time_indices_floor[t_new]
            t_ceil = time_indices_ceil[t_new]
            weight = time_weights[t_new]
            
            if t_floor == t_ceil:
                frame = tensor[:, t_floor, :, :]
            else:
                frame = (1 - weight) * tensor[:, t_floor, :, :] + weight * tensor[:, t_ceil, :, :]
            tensor_new.append(frame)
        
        return torch.stack(tensor_new, dim=1)

    def forward_vace_dual(self, x, vace_context1, vace_context2, seq_len, kwargs):
        """
        Dual vace forward: process two conditions separately then fuse hints
        """
        # embeddings for first condition
        c1 = [self.model.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context1]
        c1 = [u.flatten(2).transpose(1, 2) for u in c1]
        c1 = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c1
        ])

        # embeddings for second condition  
        c2 = [self.model.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context2]
        c2 = [u.flatten(2).transpose(1, 2) for u in c2]
        c2 = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c2
        ])

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        # process first condition through vace_blocks
        for block in self.model.vace_blocks:
            c1 = block(c1, **new_kwargs)
        hints1 = torch.unbind(c1)[:-1]

        # process second condition through vace_blocks
        for block in self.model.vace_blocks:
            c2 = block(c2, **new_kwargs)
        hints2 = torch.unbind(c2)[:-1]

        return hints1, hints2

    def process_mask_to_latent_space(self, original_mask, target_frames, target_h, target_w, dilate_pixels=10, blur_sigma=4.0):
        """
        Process original mask to latent space dimensions
        eg:original_mask: [1, 21, 512, 512] -> [1, 7, 64, 64]
        """
        mask = original_mask.clone()  
        
        if dilate_pixels > 0 or blur_sigma > 0.01:
            mask = self.process_mask_dilate_blur(mask, dilate_pixels, blur_sigma)
        
        if mask.shape[1] != target_frames - 1:  
            mask = self._temporal_interpolation(mask, target_frames - 1)
        
        
        ref_mask = torch.zeros_like(mask[:, :1, :, :])  # [1, 1, H, W]
        mask = torch.cat([ref_mask, mask], dim=1)  # [1, target_frames, H, W]
        
       
        if mask.shape[-2:] != (target_h, target_w):
            mask = F.interpolate(mask, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return mask
    def new_unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """
        
        c = 1536 // 4
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
    
    def fuse_hints_with_mask(self, hints1, hints2, original_mask, grid_sizes, dilate_pixels=10, blur_sigma=4.0):
        """
        Fuse two sets of hints using spatial fusion approach:
        1. Unpatch hints to spatial form
        2. Process mask to match latent dimensions  
        3. Fuse in spatial domain
        4. Re-patch to sequence form
        """
        # hints shape: [batch_size, seq_len, dim]
        # grid_sizes: [batch_size, 3] where 3 = [F, H, W] patches
        
        fused_hints = []
        for i, (h1, h2) in enumerate(zip(hints1, hints2)):
            # h1, h2 shape: [batch_size, seq_len, dim] 
            batch_size, seq_len, dim = h1.shape
            # grid_size = grid_sizes[i] if len(grid_sizes.shape) > 1 else grid_sizes  # [F, H, W]
            # print(f"h1.shape = {h1.shape}, grid_size.shape = {grid_sizes.shape},grid_size = {grid_sizes}") #h1.shape = torch.Size([1, 7168, 1536]), grid_size.shape = torch.Size([1, 3]),grid_size = tensor([[ 7, 32, 32]])
            # 1. Unpatch hints to spatial form
            # unpatchify expects [seq_len, dim], so we take the first batch
            h1_spatial_list = self.new_unpatchify(h1, grid_sizes) 
            h2_spatial_list = self.new_unpatchify(h2, grid_sizes)
            # print(h1_spatial_list[0].shape)  #([384, 7, 64, 64])
            h1_spatial = h1_spatial_list[0]  # [dim, F, H, W]
            h2_spatial = h2_spatial_list[0]  # [dim, F, H, W]
            
            # 2. Process mask to match spatial dimensions
            target_f, target_h, target_w = h1_spatial.shape[1:]
            processed_mask = self.process_mask_to_latent_space(
                original_mask, target_f, target_h, target_w, dilate_pixels, blur_sigma
            )  # [1, F, H, W]
            
           
            fusion_mask = processed_mask.expand(dim // 4, -1, -1, -1)  # [dim, F, H, W]
            
            # 3.fuse
            fused_spatial = (1 - fusion_mask) * h1_spatial + fusion_mask * h2_spatial
            
            # 4. Re-patch back to sequence form
            fused_sequence = self._repatch_to_sequence(fused_spatial, grid_sizes[0])  # [actual_seq_len, dim]
            
            # 5. Ensure sequence length matches original (pad or truncate if necessary)
            actual_seq_len = fused_sequence.shape[0]
            if actual_seq_len < seq_len:
                # Pad with zeros
                padding = torch.zeros(seq_len - actual_seq_len, dim, 
                                    device=fused_sequence.device, dtype=fused_sequence.dtype)
                fused_sequence = torch.cat([fused_sequence, padding], dim=0)
                print("actseq < seq-----------------")
            elif actual_seq_len > seq_len:
                # Truncate
                fused_sequence = fused_sequence[:seq_len]
                print("actseq > seq------------------")
            # 6. refresh batch dim
            fused_hint = fused_sequence.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, dim]
            fused_hints.append(fused_hint)
        
        return fused_hints
    
    def _repatch_to_sequence(self, spatial_tensor, grid_size):
        """
        Convert spatial tensor back to sequence form (exact reverse of unpatchify)
        This is a pure dimension rearrangement without any learnable parameters.
        
        spatial_tensor: [dim, F, H, W] 
        grid_size: [F_patches, H_patches, W_patches]
        
        Returns: [seq_len, dim * patch_volume] where seq_len = F_patches * H_patches * W_patches
        """
        dim, F, H, W = spatial_tensor.shape
        F_patches, H_patches, W_patches = grid_size
        patch_f, patch_h, patch_w = self.patch_size
        
        # This exactly reverses the unpatchify operations:
        # unpatchify does:
        # 1. u.view(*v, *patch_size, c) 
        # 2. einsum('fhwpqrc->cfphqwr', u)
        # 3. u.reshape(c, *[i * j for i, j in zip(v, patch_size)])
        
        # Reverse step 3: [dim, F, H, W] -> [dim, F_patches, patch_f, H_patches, patch_h, W_patches, patch_w]
        reshaped = spatial_tensor.reshape(
            dim, F_patches, patch_f, H_patches, patch_h, W_patches, patch_w
        )
        
        # Reverse step 2: 'cfphqwr->fhwpqrc' (reverse einsum)
        rearranged = torch.einsum('cfphqwr->fhwpqrc', reshaped)
        
        # Reverse step 1: flatten back to sequence form
        # [F_patches, H_patches, W_patches, patch_f, patch_h, patch_w, dim]
        # -> [F_patches * H_patches * W_patches, patch_f * patch_h * patch_w * dim]
        seq_len = F_patches * H_patches * W_patches
        patch_volume = patch_f * patch_h * patch_w
        sequence = rearranged.reshape(seq_len, patch_volume * dim)
        
        return sequence

    def generate_with_dual_conditions(self,
                                    input_prompt,
                                    input_frames1,     
                                    input_masks1,      
                                    input_frames2,     
                                    input_masks2,      
                                    input_ref_images,
                                    size=(1280, 720),
                                    frame_num=81,
                                    context_scale=1.0,
                                    shift=5.0,
                                    sample_solver='unipc',
                                    sampling_steps=50,
                                    guide_scale=5.0,
                                    n_prompt="",
                                    seed=-1,
                                    offload_model=True,
                                    dilate_pixels=10,
                                    blur_sigma=4.0,
                                    inject_depth_step=50):
        """
        Generate with dual conditions: process separately then fuse hints
        """
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

       
        z1 = self.vace_encode_frames(input_frames1, input_ref_images, masks=input_masks1)
        m1 = self.vace_encode_masks(input_masks1, input_ref_images)
        vace_context1 = self.vace_latent(z1, m1)

       
        z2 = self.vace_encode_frames(input_frames2, input_ref_images, masks=input_masks2)
        m2 = self.vace_encode_masks(input_masks2, input_ref_images)
        vace_context2 = self.vace_latent(z2, m2)

        
        original_fusion_mask = input_masks1[0]  # [1, T, H, W] 

        target_shape = list(z1[0].shape)
        target_shape[0] = int(target_shape[0] / 2)
        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

       
        def dual_forward(model_self, x, t, vace_context1, vace_context2, original_fusion_mask, context, seq_len, vace_context_scale=1.0, dilate_pixels=10, blur_sigma=4.0, clip_fea=None, y=None):
            
            device = model_self.patch_embedding.weight.device
            if model_self.freqs.device != device:
                model_self.freqs = model_self.freqs.to(device)

            
            x = [model_self.patch_embedding(u.unsqueeze(0)) for u in x]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            
            x = [u.flatten(2).transpose(1, 2) for u in x]
            seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
            assert seq_lens.max() <= seq_len
            x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

            # time embeddings
            with amp.autocast(dtype=torch.float32):
                e = model_self.time_embedding(
                    sinusoidal_embedding_1d(model_self.freq_dim, t).float())
                e0 = model_self.time_projection(e).unflatten(1, (6, model_self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32

            # context
            context_lens = None
            context = model_self.text_embedding(
                torch.stack([
                    torch.cat([u, u.new_zeros(model_self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))

            # arguments for vace processing
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=model_self.freqs,
                context=context,
                context_lens=context_lens)

            # dual vace processing
            hints1, hints2 = self.forward_vace_dual(x, vace_context1, vace_context2, seq_len, kwargs)
            
            # fuse hints using spatial approach with original mask
            fused_hints = self.fuse_hints_with_mask(hints1, hints2, original_fusion_mask, grid_sizes, dilate_pixels, blur_sigma)
            
            kwargs['hints'] = fused_hints
            kwargs['context_scale'] = vace_context_scale

            # iffusion blocks
            for block in model_self.blocks:
                x = block(x, **kwargs)

            # head
            x = model_self.head(x, e)

            # unpatchify
            x = model_self.unpatchify(x, grid_sizes)
            return [u.float() for u in x]

        from wan.modules.model import sinusoidal_embedding_1d
        bound_method = types.MethodType(dual_forward, self.model)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            for _, t in enumerate(tqdm(timesteps)):
                
                latent_model_input = latents
                timestep = [t]
                timestep = torch.stack(timestep)

                self.model.to(self.device)
                
                if _ < inject_depth_step:
                    
                    noise_pred_cond = bound_method(
                        latent_model_input, t=timestep, 
                        vace_context1=vace_context1, vace_context2=vace_context2,
                        original_fusion_mask=original_fusion_mask,
                        context=context, seq_len=seq_len, 
                        vace_context_scale=context_scale, 
                        dilate_pixels=dilate_pixels, blur_sigma=blur_sigma)[0]
                    
                    noise_pred_uncond = bound_method(
                        latent_model_input, t=timestep,
                        vace_context1=vace_context1, vace_context2=vace_context2,
                        original_fusion_mask=original_fusion_mask,
                        context=context_null, seq_len=seq_len,
                        vace_context_scale=context_scale,
                        dilate_pixels=dilate_pixels, blur_sigma=blur_sigma)[0]
                else:
                    if _ == inject_depth_step:
                        print(f"no fusion step{_}")
                    arg_c = {'context': context, 'seq_len': seq_len}
                    arg_null = {'context': context_null, 'seq_len': seq_len}
                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, vace_context=vace_context1, vace_context_scale=context_scale, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, vace_context=vace_context1, vace_context_scale=context_scale,**arg_null)[0]
                
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.decode_latent(x0, input_ref_images)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
