import boto3
import json
import os

def retrieve_and_generate(user_query):
    bedrock_agent = boto3.client('bedrock-agent-runtime', 
                                region_name='us-east-1') 
    
    try:
        # Call the retrieve_and_generate API
        response = bedrock_agent.retrieve_and_generate(
            input={
                'text': user_query
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    # Replace with your knowledge base ID
                    'knowledgeBaseId': 'XYZ-123',
                    # Using Nova Pro model
                    'modelArn': 'arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0'
                }
            }
        )
        
        # Extract the generated text from the response
        if 'output' in response and 'text' in response['output']:
            return response['output']['text']
        else:
            return "No response generated"
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return f"Error: {str(e)}"

def main():
    sample_queries = [
        "A relevant question about the information from your data source?"
    ]
    
    for query in sample_queries:
        print("\nQuery:", query)
        print("-" * 50)
        response = retrieve_and_generate(query)
        print("Response:", response)
        print("-" * 50)

if __name__ == "__main__":
    main()
