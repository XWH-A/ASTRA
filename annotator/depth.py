import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from .utils import convert_to_numpy,read_video_frames, save_one_video

class DepthV2Annotator:
    def __init__(self, cfg, device=None):
        from .depth_anything_v2.dpt import DepthAnythingV2
        pretrained_model = cfg['PRETRAINED_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]).to(self.device)
        self.model.load_state_dict(
            torch.load(
                pretrained_model,
                map_location=self.device
            )
        )
        self.model.eval()

    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        image = convert_to_numpy(image)
        depth = self.model.infer_image(image)

        depth_pt = depth.copy()
        depth_pt -= np.min(depth_pt)
        depth_pt /= np.max(depth_pt)
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

        depth_image = depth_image[..., np.newaxis]
        depth_image = np.repeat(depth_image, 3, axis=2)
        return depth_image


def process_video_to_depth(video_path, save_path, pretrained_model_path, device=None, keep_fps=True):
    import os
    
    cfg = {
        'PRETRAINED_MODEL': pretrained_model_path
    }
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    annotator = DepthV2Annotator(cfg, device=device)
    actual_video_path = video_path.split(",")[0]
    frames, fps, width, height, num_frames = read_video_frames(
        actual_video_path, 
        use_type='cv2', 
        info=True
    )
    
    depth_frames = []
    for i, frame in enumerate(tqdm(frames, desc="processing")):
        depth_frame = annotator.forward(np.array(frame))
        depth_frames.append(depth_frame)
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    output_fps = fps if keep_fps else 16
    save_one_video(save_path, depth_frames, fps=output_fps)
    
    
    return depth_frames