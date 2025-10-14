from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import gc

_model = None
_processor = None

def load_model(model_path="ckpt/Qwen2.5-VL-32B-Instruct"):
    global _model, _processor
    
    if _model is None:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        _processor = AutoProcessor.from_pretrained(model_path)
    
    return _model, _processor
def unload_model():
    global _model, _processor
    
    if _model is not None:
        del _model
        _model = None
    
    if _processor is not None:
        del _processor
        _processor = None
    
 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    
    gc.collect()
    

def enhance_prompt(original_prompt, reference_image=None, model_path="ckpt/Qwen2.5-VL-32B-Instruct"):
    

    system_prompt = """You are a prompt engineer. Your task is to rewrite user input into high-quality prompts for better video generation, ensuring that the original meaning and context are maintained while enhancing the description.
Requirements:
1.For user inputs with subjects and reference images, expand the subject's appearance features based on the reference image, focusing on texture, shape, and appearance, but ensure that any subtle action or pose from the original sentence is integrated. This ensures the subject's dynamic presence is maintained.
2.Enhance main features (appearance, expression, pose), visual style, and camera scale based on the subject's visual reference image, incorporating any motion or action present in the original description.
3.Expansions should follow the original sentence, adding rich detail about the subject's physical characteristics and actions without altering the meaning or tone. Ensure the subject’s motion or context is not lost.
4.Focus on the object's appearance and action, adding subtle but impactful context to enrich the scene.
5.Output the entire prompt in English, preserving the original sentence and context while enhancing the object’s details, and keeping any action or pose information intact.
6.The prompt length should be around 50-80 words, accurately reflecting user intent with detailed, rich object descriptions. The prompt word should be a complete sentence, with a focus on dynamic and descriptive actions as part of the visual scene.

Example Adjustments for Action Integration:
Original:
"Three robots dancing energetically in a fitness studio."
Updated:
"Three humanoid robots with sleek, high-gloss exoskeletons—primarily pearl-white with deep graphite inlays—dance energetically in a mirrored fitness studio. Each robot features an elongated, humanlike frame with segmented plating that articulates smoothly at the joints. Their oval heads have expressionless faces with luminous cyan eyes and integrated auditory sensors on each side. Chest panels reveal intricate mechanical cores beneath transparent plating, softly aglow with pulsing circuitry. Their articulated fingers and reinforced lower limbs move with astonishing precision, enhanced by jointed servos and cushioned footplates that grip the floor rhythmically. The contrast of sleek metal and fluid motion lends an uncanny vitality."

Examples:
"Three robots dancing energetically in a fitness studio."
→ "Three humanoid robots with sleek, high-gloss exoskeletons—primarily pearl-white with deep graphite inlays—dance energetically in a mirrored fitness studio. Each robot features an elongated, humanlike frame with segmented plating that articulates smoothly at the joints. Their oval heads have expressionless faces with luminous cyan eyes and integrated auditory sensors on each side. Chest panels reveal intricate mechanical cores beneath transparent plating, softly aglow with pulsing circuitry. Their articulated fingers and reinforced lower limbs move with astonishing precision, enhanced by jointed servos and cushioned footplates that grip the floor rhythmically. The contrast of sleek metal and fluid motion lends an uncanny vitality."
2. "Eight Iron Men leap over a purple obstacle in a race"
→ Eight Iron Men, clad in vibrant crimson and gold armor, leap over a purple obstacle in a race, their armor gleaming in the light. Segmented armor hugs their muscular physiques, accentuating their sharp, angular features. Their sleek helmets, adorned with sharp eye slits, glow a cold white, even more clearly against the rich crimson of their armor. Each Iron Man's arc reactor glows blue from his chest, embedded in the chestplate and intricately detailed. Their hydraulically articulated limbs propel them gracefully and synchronously over the purple obstacle, adding a sense of dynamism to the scene. Wide-angle shots emphasize the action.
Now rewrite the following prompt directly in English, preserving original semantics and actions while adding more details to the subject description by referring to the input image."""

    try:
        model, processor = load_model(model_path)
        

        user_content = []

        if reference_image and os.path.exists(reference_image):
            user_content.append({
                "type": "image",
                "image": reference_image
            })
        
        user_content.append({
            "type": "text",
            "text": original_prompt
        })
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        result = output_text[0].strip()
        
        return result
        
    except Exception as e:
        print(f"Enhancement failed: {e}")
        return original_prompt
    finally:
        unload_model()


if __name__ == "__main__":
    enhanced2 = enhance_prompt("Two iron-men jumping rope.", "/opt/data/private/video_edit/VACE_formal/MSED/ref_images/Iron Man.png",)
    print("Enhanced:", enhanced2)


