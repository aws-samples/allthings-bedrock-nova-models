import boto3
import base64
import json
from PIL import Image
import io

def nova_canvas_segmentation_example():
    try:
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
        with open('./images/generated_image.png', 'rb') as image_file:
            condition_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        request_body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": "3d animated film style cat",
                "conditionImage": condition_image,
                "controlMode": "SEGMENTATION",
                "controlStrength": 0.7
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "standard",
                "cfgScale": 8.0
            }
        }

        response = client.invoke_model(
            modelId="amazon.nova-canvas-v1:0",
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response['body'].read())
        image_data = base64.b64decode(response_body['images'][0])
        image = Image.open(io.BytesIO(image_data))
        image.save('./images/generated_segmentation.png')
        print("Generated image saved as: ./images/generated_segmentation.png")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    nova_canvas_segmentation_example()
