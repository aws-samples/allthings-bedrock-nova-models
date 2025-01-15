import boto3
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

class NovaConverse:
    def __init__(self, model_id: str = "amazon.nova-pro-v1:0", region: str = "us-east-1"):
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.logger = logging.getLogger(__name__)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def create_message(self, 
                      text: str, 
                      image_path: Optional[str] = None,
                      role: str = "user") -> Dict[str, Any]:
        """Create a message for the conversation."""
        content = []
        
        if text:
            content.append({"text": text})
        
        return {
            "role": role,
            "content": content
        }

    def converse(self, 
                 messages: list,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 max_tokens: int = 2048,
                 stop_sequences: list = None,
                 stream: bool = False) -> Dict[str, Any]:
        try:
            inference_config = {
                "temperature": temperature,
                "topP": top_p,
                "maxTokens": max_tokens
            }
            
            if stop_sequences:
                inference_config["stopSequences"] = stop_sequences

            request_params = {
                "modelId": self.model_id,
                "messages": messages,
                "inferenceConfig": inference_config  
            }

          
            response = self.bedrock.converse(**request_params)
            
            return response

        except Exception as e:
            self.logger.error(f"Error in conversation: {str(e)}")
            raise

def text_conversation_example():
    """Example of a text-based conversation."""
    nova = NovaConverse()
    
    messages = [
        nova.create_message(
            "What are the main differences between supervised and unsupervised learning?"
        )
    ]
    
    response = nova.converse(messages)
    
    if response and "output" in response:
        assistant_message = response["output"]["message"]
        print("\nAssistant's response:")
        print(assistant_message["content"][0]["text"])
        
        messages.append(assistant_message)
        messages.append(
            nova.create_message(
                "Can you provide a specific example of each?"
            )
        )
        
        follow_up_response = nova.converse(messages)
        if follow_up_response and "output" in follow_up_response:
            print("\nFollow-up response:")
            print(follow_up_response["output"]["message"]["content"][0]["text"])
            
            usage = follow_up_response.get("usage", {})
            print("\nToken Usage:")
            print(f"Input tokens: {usage.get('inputTokens', 0)}")
            print(f"Output tokens: {usage.get('outputTokens', 0)}")

def main():
    try:
        print("Running text conversation example...")
        text_conversation_example()
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
