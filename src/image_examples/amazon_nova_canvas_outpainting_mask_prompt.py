import base64
import json
import os
import boto3
import random
from PIL import Image
import io

def outpaint_with_mask_prompt(input_image_path, prompt, mask_prompt):
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    with open(input_image_path, "rb") as image_file:
        input_image = base64.b64encode(image_file.read()).decode('utf-8')

    request_body = {
        "taskType": "OUTPAINTING",
        "outPaintingParams": {
            "text": prompt,
            "image": input_image,
            "maskPrompt": mask_prompt,
            "outPaintingMode": "PRECISE",
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "height": 1024,
            "width": 1024,
            "cfgScale": 8.0,
            "seed": random.randint(1, 2147483647),
        }
    }

    try:
        response = client.invoke_model(
            modelId="amazon.nova-canvas-v1:0",
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response.get("body").read())
    
        if "images" in response_body and len(response_body["images"]) > 0:
            base64_image = response_body["images"][0]
            image_bytes = base64.b64decode(base64_image)
            
            output_path = './images/outpainted_image_mask_prompt.png'
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            
            print(f"Generated outpainted image saved to: {output_path}")
            return output_path

        else:
            print("No image was generated in the response")
            return None

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

if __name__ == "__main__":
    input_image_path = "./images/generated_image.png"
    mask_prompt = "cat"  # Areas to keep unchanged
    prompt = "a beautiful garden in the background" 
    result = outpaint_with_mask_prompt(input_image_path, prompt, mask_prompt)
