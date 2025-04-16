"""This script is trying to test vision models availability from DeepSeek API key using LiteLLM."""

import os 
from litellm import completion

api_key = os.getenv('DEEPSEEK_API_KEY')

# openai call
response = completion(
    model = "deepseek-vl-1.3b-chat", 
    messages=[
        {
            "role": "user",
            "content": [
                            {
                                "type": "text",
                                "text": "What’s in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                }
                            }
                        ]
        }
    ],
)

import litellm
assert litellm.supports_vision(model="deepseek/deepseek-vl-1.3b-chat") == True
