import os
from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
import numpy as np
import cv2



def load_grounding_dino():
    config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = "pt/groundingdino_swint_ogc.pth"
    model = load_model(config_path, weights_path)
    return model


def get_boxes(model, image, prompt, box_thresh=0.40, text_thresh=0.25):

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to("cuda") / 255.0
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        device="cuda"
    )
    
    h, w = image.shape[:2]
    boxes_xyxy = boxes * torch.tensor([w, h, w, h])
    return boxes_xyxy.cpu().numpy(), (w, h)

def compute_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def compute_CM_Err(source_boxes, edit_boxes, image_size):
    w, h = image_size
    D_max = np.sqrt(w ** 2 + h ** 2) 

    n1 = len(source_boxes)
    n2 = len(edit_boxes)

    matched = []
    used_src = set()
    used_edit = set()

   
    distances = []
    for i, box1 in enumerate(source_boxes):
        c1 = compute_center(box1)
        for j, box2 in enumerate(edit_boxes):
            c2 = compute_center(box2)
            dist = np.linalg.norm(c1 - c2)
            distances.append((dist, i, j))
    distances.sort()  

    for dist, i, j in distances:
        if i not in used_src and j not in used_edit:
            matched.append(dist / D_max)  
            used_src.add(i)
            used_edit.add(j)

    matched_count = len(matched)
    unmatched_count = abs(n1 - n2)
    total_cost = sum(matched) + unmatched_count * 1.0  

    LD = total_cost / (matched_count + unmatched_count) if (matched_count + unmatched_count) > 0 else 0.0
    return LD, matched_count, unmatched_count

def process_videos(video_src_path, video_edt_path, prompt_src, prompt_edt, box_thresh=0.40, text_thresh=0.25, save_frames=False, output_dir="annotated_frames"):
    model = load_grounding_dino()

    cap_src = cv2.VideoCapture(video_src_path)
    cap_edt = cv2.VideoCapture(video_edt_path)

    if not cap_src.isOpened() or not cap_edt.isOpened():
        raise ValueError("无法打开一个或两个视频文件")

    width_src = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_src = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_edt = int(cap_edt.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_edt = int(cap_edt.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width_src != 512 or height_src != 512 or width_edt != 512 or height_edt != 512:
        raise ValueError(f"视频分辨率不是512x512：源视频 {width_src}x{height_src}，编辑视频 {width_edt}x{height_edt}")

    frame_count_src = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_edt = int(cap_edt.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(frame_count_src, frame_count_edt)
    if frame_count == 0:
        raise ValueError("视频没有有效帧")

    if save_frames:
        os.makedirs(output_dir, exist_ok=True)

    CM_Err_scores = []
    frame_idx = 0

    while frame_idx < frame_count:
        ret_src, frame_src = cap_src.read()
        ret_edt, frame_edt = cap_edt.read()

        if not ret_src or not ret_edt:
            break

        frame_src_rgb = cv2.cvtColor(frame_src, cv2.COLOR_BGR2RGB)
        frame_edt_rgb = cv2.cvtColor(frame_edt, cv2.COLOR_BGR2RGB)

        try:
            boxes_src, size_src = get_boxes(model, frame_src_rgb, prompt_src, box_thresh, text_thresh)
            boxes_edt, size_edt = get_boxes(model, frame_edt_rgb, prompt_edt, box_thresh, text_thresh)

            if size_src != size_edt:
                continue

            LD, matched, unmatched = compute_CM_Err(boxes_src, boxes_edt, size_src)
            CM_Err_scores.append(LD)

            if save_frames:
                
                _, image_tensor = load_image(frame_src_rgb)
                boxes, logits, phrases = predict(model, image_tensor, prompt_src, box_thresh, text_thresh, device="cuda")
                annotated_src = annotate(frame_src_rgb, boxes, logits, phrases)
                cv2.imwrite(os.path.join(output_dir, f"src_frame_{frame_idx:04d}.jpg"), cv2.cvtColor(annotated_src, cv2.COLOR_RGB2BGR))

                _, image_tensor = load_image(frame_edt_rgb)
                boxes, logits, phrases = predict(model, image_tensor, prompt_edt, box_thresh, text_thresh, device="cuda")
                annotated_edt = annotate(frame_edt_rgb, boxes, logits, phrases)
                cv2.imwrite(os.path.join(output_dir, f"edt_frame_{frame_idx:04d}.jpg"), cv2.cvtColor(annotated_edt, cv2.COLOR_RGB2BGR))

        except Exception as e:
            continue

        frame_idx += 1

    cap_src.release()
    cap_edt.release()

    if CM_Err_scores:
        avg_CM_Err = np.mean(CM_Err_scores)
        print(f"\nAvg CM_Err = {avg_CM_Err:.4f} | Total Frames: {len(CM_Err_scores)}")
        return avg_CM_Err, len(CM_Err_scores)
    else:
        return 0.0, 0

# === 主程序 ===
if __name__ == "__main__":
    video_src_path = "" #ori video
    video_edt_path = "" #edit video 

    prompt_src = "" #ori subject
    prompt_edt = "" #edit subject

    avg_ld, total_frames = process_videos(
        video_src_path,
        video_edt_path,
        prompt_src,
        prompt_edt,
    )