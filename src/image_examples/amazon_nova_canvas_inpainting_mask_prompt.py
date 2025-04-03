import boto3
import json
import base64
from PIL import Image
import io
import random

def inpaint_image(input_image_path, prompt, mask_prompt):
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    model_id = 'amazon.nova-canvas-v1:0'
    
    with open(input_image_path, "rb") as image_file:
        input_image = base64.b64encode(image_file.read()).decode('utf8')
    
    request_body = {
        "taskType": "INPAINTING",
        "inPaintingParams": {
            "text": prompt,
            "image": input_image,
            "maskPrompt": mask_prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "height": 512,
            "width": 512,
            "cfgScale": 8.0,
            "seed": random.randint(1, 2147483647),
        }
    }
    
    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response['body'].read())
        base64_image = response_body['images'][0]
        
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save the inpainted image in the images directory
        output_path = './images/inpainted_image_mask_prompt.png'
        image.save(output_path)
        print(f"Image inpainted successfully and saved as: {output_path}")
        
    except Exception as e:
        print(f"Error during inpainting: {str(e)}")

def main():
    # Example usage
    input_image_path = "./images/generated_image.png"  # Path to your source image
    prompt = "dog with a cool hat"  # Describe what you want to add/modify
    mask_prompt = "cat"  # Specify what part of the image to modify
    
    inpaint_image(input_image_path, prompt, mask_prompt)

if __name__ == "__main__":
    main()
