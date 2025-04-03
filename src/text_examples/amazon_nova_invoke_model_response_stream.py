import boto3
import json
from datetime import datetime

# Create a Bedrock Runtime client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Nova Lite model ID
NOVA_MODEL_ID = "us.amazon.nova-lite-v1:0"


system_prompts = [
    {
        "text": "You are a helpful AI assistant that provides clear and concise responses."
    }
]

messages = [
    {
        "role": "user", 
        "content": [
            {
                "text": "What are the benefits of cloud computing?"
            }
        ]
    }
]

inference_config = {
    "max_new_tokens": 500,
    "top_p": 0.9,
    "top_k": 20,
    "temperature": 0.7
}

request_body = {
    "schemaVersion": "messages-v1",
    "messages": messages,
    "system": system_prompts,
    "inferenceConfig": inference_config
}

try:
    start_time = datetime.now()
    print("Sending request...")
    response = client.invoke_model_with_response_stream(
        modelId=NOVA_MODEL_ID,
        body=json.dumps(request_body)
    )

    # Get request ID for tracking
    request_id = response.get("ResponseMetadata", {}).get("RequestId")
    print(f"Request ID: {request_id}")
    print("Response:")

    stream = response.get("body")
    if stream:
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                chunk_data = json.loads(chunk.get("bytes").decode())
                content_block = chunk_data.get("contentBlockDelta")
                if content_block:
                    # Print the text from the response
                    print(content_block.get("delta", {}).get("text", ""), end="")
    else:
        print("No response stream received")

except Exception as e:
    print(f"Error: {str(e)}")
