import argparse
import time
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.utils.utils import cache_video, cache_image, str2bool

from models.wan import WanVaceDual
from models.wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate video with dual condition processing using WanVaceDual"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vace-1.3B",
        choices=list(WAN_CONFIGS.keys()),
        help="The model name to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="480p",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to sample. The number should be 4n+1")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default='/opt/data/private/video_edit/VACE_formal/models/Wan2.1-VACE-1.3B/',
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    
    parser.add_argument(
        "--src_video1",
        type=str,
        default=False,
        help="The file of the first source video (masked video).")
    parser.add_argument(
        "--src_mask1",
        type=str,
        default=False,
        help="The file of the first source mask.")
    parser.add_argument(
        "--src_video2",
        type=str,
        default=False,
        help="The file of the second source video (depth condition).")
    parser.add_argument(
        "--src_mask2",
        type=str,
        default=None,
        help="The file of the second source mask (optional, can be None).")
    
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','.")
    
    parser.add_argument(
        "--dilate_pixels",
        type=int,
        default=10,
        help="Dilation pixels for soft mask processing based on first group mask.")
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=4.0,
        help="Gaussian blur sigma for soft mask processing based on first group mask.")
    
    # Standard parameters
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=2025,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", 
        type=int, 
        default=50, 
        help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=5.0,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--inject_depth_step",
        type=int,
        default=30,
        help="the inject_depth_step")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The directory to save the generated video.")
    
    return parser


def validate_args(args):
   
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.model_name in WAN_CONFIGS, f"Unsupport model name: {args.model_name}"

  
    assert os.path.exists(args.src_video1), f"src_video1 file not found: {args.src_video1}"
    assert os.path.exists(args.src_mask1), f"src_mask1 file not found: {args.src_mask1}"
    assert os.path.exists(args.src_video2), f"src_video2 file not found: {args.src_video2}"
    
    if args.src_mask2 is not None:
        assert os.path.exists(args.src_mask2), f"src_mask2 file not found: {args.src_mask2}"

    if args.src_ref_images is not None:
        ref_images = args.src_ref_images.split(',')
        for ref_img in ref_images:
            if ref_img.strip() and not os.path.exists(ref_img.strip()):
                print(f"Warning: reference image not found: {ref_img.strip()}")

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    
    
    assert args.size in SUPPORTED_SIZES[args.model_name], \
        f"Unsupport size {args.size} for model name {args.model_name}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.model_name])}"
    
    return args


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def main(args):
    args = argparse.Namespace(**args) if isinstance(args, dict) else args
    args = validate_args(args)

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS[args.model_name]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]
    
    logging.info("Creating enhanced WanVace pipeline with dual condition processing.")
    wan_vace_dual = WanVaceDual(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    logging.info("Preparing first group inputs (masked video)...")
    src_video1, src_mask1, src_ref_images1 = wan_vace_dual.prepare_source(
        [args.src_video1],
        [args.src_mask1],
        [None if args.src_ref_images is None else args.src_ref_images.split(',')],
        args.frame_num, 
        SIZE_CONFIGS[args.size], 
        device
    )

    logging.info("Preparing second group inputs (depth condition)...")
    src_video2, src_mask2, src_ref_images2 = wan_vace_dual.prepare_source(
        [args.src_video2],
        [args.src_mask2],  
        [None if args.src_ref_images is None else args.src_ref_images.split(',')],
        args.frame_num, 
        SIZE_CONFIGS[args.size], 
        device
    )

    src_ref_images = src_ref_images1

    logging.info(f"Generating video with dual condition processing...")
    logging.info(f"First group video shape: {src_video1[0].shape}")
    logging.info(f"First group mask shape: {src_mask1[0].shape}")
    logging.info(f"Second group video shape: {src_video2[0].shape}")
    logging.info(f"Second group mask shape: {src_mask2[0].shape}")
    
    video = wan_vace_dual.generate_with_dual_conditions(
        args.prompt,
        src_video1,          
        src_mask1,           
        src_video2,          
        src_mask2,           
        src_ref_images,      
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
        dilate_pixels=args.dilate_pixels,
        blur_sigma=args.blur_sigma,
        inject_depth_step=args.inject_depth_step
    )

    ret_data = {}
    if rank == 0:
        if args.save_dir is None:
            save_dir = os.path.join('results', 'dual_conditions', args.model_name, 
                                   time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
        else:
            save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        
        save_file = os.path.join(save_dir, 'dual_conditions_result.mp4')
        cache_video(
            tensor=video[None],
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        logging.info(f"Saving generated video to {save_file}")
        ret_data['out_video'] = save_file

        # Save source videos and masks for reference
        save_file = os.path.join(save_dir, 'src_video1_masked.mp4')
        cache_video(tensor=src_video1[0][None], save_file=save_file,
                   fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
        logging.info(f"Saving src_video1 (masked) to {save_file}")
        ret_data['src_video1'] = save_file

        save_file = os.path.join(save_dir, 'src_mask1.mp4')
        cache_video(tensor=src_mask1[0][None], save_file=save_file,
                   fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(0, 1))
        logging.info(f"Saving src_mask1 to {save_file}")
        ret_data['src_mask1'] = save_file

        save_file = os.path.join(save_dir, 'src_video2_depth.mp4')
        cache_video(tensor=src_video2[0][None], save_file=save_file,
                   fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
        logging.info(f"Saving src_video2 (depth) to {save_file}")
        ret_data['src_video2'] = save_file

        save_file = os.path.join(save_dir, 'src_mask2.mp4')
        cache_video(tensor=src_mask2[0][None], save_file=save_file,
                   fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(0, 1))
        logging.info(f"Saving src_mask2 to {save_file}")
        ret_data['src_mask2'] = save_file

        # Save reference images if provided
        if src_ref_images[0] is not None:
            for i, ref_img in enumerate(src_ref_images[0]):
                save_file = os.path.join(save_dir, f'src_ref_image_{i}.png')
                cache_image(
                    tensor=ref_img[:, 0, ...],
                    save_file=save_file,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))
                logging.info(f"Saving src_ref_image_{i} to {save_file}")
                ret_data[f'src_ref_image_{i}'] = save_file
        info_file = os.path.join(save_dir, 'generation_info.txt')
        with open(info_file, 'w') as f:
            f.write(f"Generation Info\n")
            f.write(f"================\n")
            f.write(f"Prompt: {args.prompt}\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Size: {args.size}\n")
            f.write(f"Frame num: {args.frame_num}\n")
            f.write(f"Sample steps: {args.sample_steps}\n")
            f.write(f"Guide scale: {args.sample_guide_scale}\n")
            f.write(f"Seed: {args.base_seed}\n")
            f.write(f"Dilate pixels: {args.dilate_pixels}\n")
            f.write(f"Blur sigma: {args.blur_sigma}\n")
            f.write(f"Source video 1: {args.src_video1}\n")
            f.write(f"Source mask 1: {args.src_mask1}\n")
            f.write(f"Source video 2: {args.src_video2}\n")
            f.write(f"Source mask 2: {args.src_mask2}\n")
            f.write(f"Reference images: {args.src_ref_images}\n")
        logging.info(f"Saving generation info to {info_file}")

    logging.info("Dual condition processing generation finished.")
    return ret_data


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    main(args)