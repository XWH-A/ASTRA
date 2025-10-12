python pipeline.py \
    --ori_video "data/ori_video/110m-Hurdles-Mid-Air-Dash.mp4" \
    --mask_video "data/mask_video/110m-Hurdles-Mid-Air-Dash.mp4" \
    --prompt "Eight Iron Men leap mid-race over purple hurdles." \
    --subject "Iron-Man" \
    --depth_model_path "ckpt/depth/depth_anything_v2_vitl.pth" \
    --sdxl_model_path "ckpt/stable-diffusion-xl-base-1.0" \
    --vlm_model_path "ckpt/Qwen2.5-VL-32B-Instruct" \
    --ckpt_dir "ckpt/Wan2.1-VACE-1.3B"