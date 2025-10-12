import argparse
import os
import sys
from annotator.depth import process_video_to_depth
from annotator.vl_extend_prompts import enhance_prompt
from annotator.utils import overlay_mask_on_video
from diffusers import DiffusionPipeline
import torch
from Inference import get_parser as get_inference_parser, main as inference_main


def get_parser():
    inference_parser = get_inference_parser()
    parser = argparse.ArgumentParser(
        description="Video editing pipeline with preprocessing and inference",
        parents=[inference_parser],
        add_help=False,
        conflict_handler='resolve'  
    )
    
    pipeline_group = parser.add_argument_group('Pipeline specific arguments (required)')
    pipeline_group.add_argument(
        "--ori_video",
        type=str,
        required=True,
        help="Path to the original video file."
    )
    pipeline_group.add_argument(
        "--mask_video",
        type=str,
        required=True,
        help="Path to the mask video file."
    )
    pipeline_group.add_argument(
        "--subject",
        type=str,
        required=True,
        help="Subject name for reference image generation (e.g., 'Iron-Man')."
    )
    
    preprocess_group = parser.add_argument_group('Preprocessing model paths (required)')
    preprocess_group.add_argument(
        "--depth_model_path",
        type=str,
        required=True,
        help="Path to depth estimation model (e.g., 'ckpt/depth/depth_anything_v2_vitl.pth')."
    )
    preprocess_group.add_argument(
        "--sdxl_model_path",
        type=str,
        required=True,
        help="Path to SDXL model for reference image generation."
    )
    preprocess_group.add_argument(
        "--vlm_model_path",
        type=str,
        required=True,
        help="Path to VLM model for prompt enhancement (e.g., Qwen2.5-VL)."
    )
    preprocess_optional = parser.add_argument_group('Preprocessing optional arguments')
    preprocess_optional.add_argument(
        "--keep_fps",
        type=bool,
        default=True,
        help="Whether to keep original video FPS. Default: True"
    )
    parser.add_argument(
        '-h', '--help',
        action='help',
        help='Show this help message and exit'
    )
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    pre_save_dir = "results/preprocess"
    
    video_basename = os.path.splitext(os.path.basename(args.ori_video))[0]
    
    print("\n[Step 1/3] Generating depth video...")
    depth_save_dir = f"{pre_save_dir}/depth"
    os.makedirs(depth_save_dir, exist_ok=True)
    depth_video_path = f"{depth_save_dir}/{video_basename}.mp4"
    
    depth_frames = process_video_to_depth(
        video_path=args.ori_video,
        save_path=depth_video_path,
        pretrained_model_path=args.depth_model_path,
        device="cuda",
        keep_fps=args.keep_fps
    )
    del depth_frames
    
    print("\n[Step 2/3] Generating reference image...")
    img_save_dir = f"{pre_save_dir}/ref-image"
    os.makedirs(img_save_dir, exist_ok=True)
    ref_image_path = f"{img_save_dir}/{args.subject}.png"
    
    if os.path.exists(ref_image_path):
        pass
    else:
        pipe = DiffusionPipeline.from_pretrained(
            args.sdxl_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        pipe.to("cuda")
        
        t2i_prompt = f"A photo of {args.subject}, full body, standing pose, cinematic lighting"
        image = pipe(prompt=t2i_prompt).images[0]
        image.save(ref_image_path)
        
        del pipe
        torch.cuda.empty_cache()
    
    print("\n[Step 3/3] Enhancing prompt with VLM...")
    print(f"Original prompt: {args.prompt}")
    enhanced_prompt = enhance_prompt(
        original_prompt=args.prompt,
        reference_image=ref_image_path,
        model_path=args.vlm_model_path
    )
    print(f"Enhanced prompt: {enhanced_prompt}")
    
    src_save_dir = f"{pre_save_dir}/src"
    os.makedirs(src_save_dir, exist_ok=True)
    src_video_path = f"{src_save_dir}/{video_basename}.mp4"
    
    overlay_mask_on_video(
        original_video_path=args.ori_video,
        mask_video_path=args.mask_video,
        save_path=src_video_path,
        keep_fps=args.keep_fps
    )
    
    print("\n" + "=" * 80)
    print("Video Editing Pipeline - Inference Stage")
    print("=" * 80)
    
    
    args.src_video1 = src_video_path       
    args.src_mask1 = args.mask_video        
    args.src_video2 = depth_video_path      
    args.src_mask2 = None                  
    args.prompt = enhanced_prompt           
    args.src_ref_images = ref_image_path    
    result = inference_main(args)
    
    return result


if __name__ == "__main__":
    main()