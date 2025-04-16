import asyncio
import base64
import json
import os
import requests
from typing import Optional


API_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAME = "gpt-4o"  # Using OpenAI multimodal models (multimodal)
MAX_TOKENS = 1500
TEMPERATURE = 0.5
TOP_P = 0.5


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Reads an image from disk and returns its base64 encoded string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"[ERROR] Image file not found: {image_path}")
    except Exception as e:
        print(f"[ERROR] Failed to read or encode image: {e}")
    return None


def build_messages(prompt: str, base64_image: str) -> list:
    """Constructs the chat messages in DeepSeek-VL compatible format."""
    return [
        {
            "role": "system",
            "content": (
                "You are ChatPal, an AI assistant powered by DeepSeek Vision Model, with computer vision.\n\n"
                "Built-in vision capabilities:\n"
                "- extract text from image\n"
                "- describe images\n"
                "- analyze image contents"
            ),
        },
        {
            "role": "user",
            "content": prompt,  # Text prompt
        },
        {
            "role": "user",
            "content": f"data:image/jpeg;base64,{base64_image}",  # Image data
        }
    ]


def build_request_payload(messages: list) -> dict:
    """Creates the payload to send to the DeepSeek API."""
    return {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "temperature": TEMPERATURE,
    }


async def reparse_record_from_image(ss_path: str, input_record: dict, retries: int = 3) -> Optional[str]:
    """Attempts to reparse a record from an image using DeepSeek Vision API."""
    for attempt in range(1, retries + 1):
        print(f"[INFO] Attempt {attempt} of {retries}...")

        base64_image = encode_image_to_base64(ss_path)
        if not base64_image:
            return None

        prompt = (
            f"You are tasked with understanding the content of {json.dumps(input_record)} and image.\n"
            "You need to reparse the record and rewrite it filling the missing values.\n"
            "Do not assume anything or hallucinate. Just fill the missing value with text extracted from image. Do not add any extra information." \
            "Try to use the **BUILT-IN OCR** capabilities of the model to extract text from the image.\n"
        )

        messages = build_messages(prompt, base64_image)
        payload = build_request_payload(messages)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                print(f"[SUCCESS] Response received:\n{result}")
                return result
            else:
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[ERROR] Exception during API call: {e}")

    print("[FAILURE] All retry attempts failed.")
    return None


if __name__ == "__main__":
    # Example usage
    screenshot_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/medium/item_0.png"
    record = {
        "headline": "",
        "publish_date": "",
        "number_of_views": "357",
        "sequence_number": "1"
    }

    asyncio.run(reparse_record_from_image(screenshot_path, record))