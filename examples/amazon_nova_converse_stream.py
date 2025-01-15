import boto3
import json

def converse_nova_stream():
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    model_id = "amazon.nova-pro-v1:0"
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "text": "Explain briefly the benefits of Amazon S3"
                }
            ]
        }
    ]

    try:
        # Call converse_stream API
        streaming_response = client.converse_stream(
            modelId=model_id,
            messages=conversation,
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.5,
                "topP": 0.9
            }
        )

        # Process the streaming response
        print("\nStreaming response:")
        stream = streaming_response.get('stream')
        if stream:
            for event in stream:
                if 'messageStart' in event:
                    print(f"\nRole: {event['messageStart']['role']}")

                if 'contentBlockDelta' in event:
                    print(event['contentBlockDelta']['delta']['text'], end="", flush=True)

                if 'messageStop' in event:
                    print(f"\nStop reason: {event['messageStop']['stopReason']}")

                if 'metadata' in event:
                    metadata = event['metadata']
                    if 'usage' in metadata:
                        print("\nToken usage:")
                        print(f"Input tokens: {metadata['usage']['inputTokens']}")
                        print(f"Output tokens: {metadata['usage']['outputTokens']}")
                        print(f"Total tokens: {metadata['usage']['totalTokens']}")

        print("\n")  

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    converse_nova_stream()
