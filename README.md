# Amazon Nova Models Examples

This repository provides a collection of Python examples demonstrating how to interact with Amazon's Nova models through Amazon Bedrock.

## Repository Structure
```
.
├── src/
│   ├── image_examples/           # Image manipulation examples using Nova Canvas
│   │   ├── amazon_nova_canvas_cannyedge.py              # Image Conditioning - Canny Edge mode
│   │   ├── amazon_nova_canvas_image_generation.py       # Basic image generation from text
│   │   ├── amazon_nova_canvas_inpainting_mask_prompt.py # Image inpainting with mask prompt
│   │   ├── amazon_nova_canvas_outpainting_mask_prompt.py # Image outpainting with mask prompt
│   │   └── amazon_nova_canvas_segmentation.py           # Image Conditioning - Segmentation mode
│   └── text_examples/            # Text generation and conversation examples
│       ├── amazon_nova_converse_stream.py               # Streaming conversation interface
│       ├── amazon_nova_converse.py                      # Basic conversation implementation
│       ├── amazon_nova_invoke_model_response_stream.py  # Streaming model responses
│       ├── amazon_nova_invoke_model.py                  # Direct model invocation
│       └── amazon_nova_retrieve_generate.py             # Knowledge base integration
```

## Usage Instructions
### Prerequisites
- Python 3.7 or higher
- AWS Account with Bedrock access
- AWS CLI configured with appropriate credentials
- Required Python packages:
  - boto3
  - Pillow (PIL)
  - json
  - base64

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install required packages
pip install boto3 Pillow

# Configure AWS credentials if not already done
aws configure
```

### Troubleshooting
Common Issues and Solutions:

1. AWS Credentials Error
```
 Ensure AWS credentials are properly configured
```

2. Model Access Issues
```
Verify your AWS region is correct and model access is provided to Amazon Nova models in Amazon Bedrock service. 
```

3. Image Generation Failures
- Check if the images directory exists
- Ensure proper file permissions in the output directory

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

## Disclaimer

You should not use this Content in your production accounts, or on production or other critical data. You are responsible for testing, securing, and optimizing content, such as sample code, as appropriate for production grade use based on your specific quality control practices and standards

