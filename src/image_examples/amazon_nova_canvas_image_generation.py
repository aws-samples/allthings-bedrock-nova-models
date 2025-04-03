import boto3
import json
import base64
import random
import os

def generate_image(prompt):
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    model_id = 'amazon.nova-canvas-v1:0'
    
    request_body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
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
        # Create images directory if it doesn't exist
        images_dir = './images'
        os.makedirs(images_dir, exist_ok=True)
        
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response['body'].read())
        base64_image = response_body['images'][0]
        image_path = os.path.join(images_dir, 'generated_image.png')
        with open(image_path, 'wb') as f:
            f.write(base64.b64decode(base64_image))
            
        print(f"Image generated successfully and saved as: {image_path}")
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")

def main():
    prompt = "A beautiful cat sitting on a bench in New York City"
    generate_image(prompt)

if __name__ == "__main__":
    main()
