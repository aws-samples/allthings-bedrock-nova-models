import boto3
import json

def invoke_nova_model():
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )

    # Define the messages for the conversation with content as an array of objects
    messages = [
        {
            "role": "user",
            "content": [{"text": "What are the key features of Amazon S3?"}]
        }
    ]

    # Prepare the request body with only supported parameters
    request_body = {
        "messages": messages
    }

    try:
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId='us.amazon.nova-lite-v1:0',
            body=json.dumps(request_body)
        )

        response_body = json.loads(response['body'].read())
        print("Full Response:", json.dumps(response_body, indent=2))

    except Exception as e:
        print(f"Error invoking model: {str(e)}")

if __name__ == "__main__":
    invoke_nova_model()
