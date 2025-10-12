
import copy
import io
import os

import torch
import numpy as np
import cv2
import imageio
from PIL import Image
import pycocotools.mask as mask_utils

def overlay_mask_on_video(original_video_path, mask_video_path, save_path, keep_fps=True):
    
    original_frames, fps, _, _, _ = read_video_frames(
        original_video_path.split(",")[0], use_type='cv2', info=True
    )
    mask_frames, _, _, _, _ = read_video_frames(
        mask_video_path.split(",")[0], use_type='cv2', info=True
    )
    
    min_frames = min(len(original_frames), len(mask_frames))
    result_frames = []
    
    for orig_frame, mask_frame in zip(original_frames[:min_frames], mask_frames[:min_frames]):
        orig_frame = np.array(orig_frame).astype(np.float32)
        mask_frame = np.array(mask_frame).astype(np.float32)
        
        if orig_frame.shape[:2] != mask_frame.shape[:2]:
            mask_frame = cv2.resize(mask_frame, (orig_frame.shape[1], orig_frame.shape[0]))
        
        mask_gray = np.mean(mask_frame, axis=2) if len(mask_frame.shape) == 3 else mask_frame
        mask_normalized = (mask_gray / 255.0)[..., np.newaxis]
        
        result = orig_frame * (1 - mask_normalized) + mask_frame * mask_normalized
        result_frames.append(np.clip(result, 0, 255).astype(np.uint8))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_one_video(save_path, result_frames, fps=fps if keep_fps else 16)
    
    # return result_frames

def convert_to_numpy(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
    return image

def save_one_video(file_path, videos, fps=8, quality=8, macro_block_size=None):
    try:
        video_writer = imageio.get_writer(file_path, fps=fps, codec='libx264', quality=quality, macro_block_size=macro_block_size)
        for frame in videos:
            video_writer.append_data(frame)
        video_writer.close()
        return True
    except Exception as e:
        print(f"Video save error: {e}")
        return False

def read_video_frames(video_path, use_type='cv2', is_rgb=True, info=False):
    frames = []
    if use_type == "decord":
        import decord
        decord.bridge.set_bridge("native")
        try:
            cap = decord.VideoReader(video_path)
            total_frames = len(cap)
            fps = cap.get_avg_fps()
            height, width, _ = cap[0].shape
            frames = [cap[i].asnumpy() for i in range(len(cap))]
        except Exception as e:
            print(f"Decord read error: {e}")
            return None
    elif use_type == "cv2":
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if is_rgb:
                    frames.append(frame[..., ::-1])
                else:
                    frames.append(frame)
            cap.release()
            total_frames = len(frames)
        except Exception as e:
            print(f"OpenCV read error: {e}")
            return None
    else:
        raise ValueError(f"Unknown video type {use_type}")
    if info:
        return frames, fps, width, height, total_frames
    else:
        return frames

