from openai import OpenAI
import os
import asyncio
import json
import base64
from typing import Optional


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


async def fill_empty_record(
        base_url: str,
        model: str,
        image_url: str,
        prompt: str, 
    ):

    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("HUGGINGFACE_API_KEY")
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": { "url": image_url},
                    }
                ]
            }
        ],
        max_tokens=1500,
    )
    print(completion.choices[0].message)

    

if __name__ == "__main__":
    # Example usage
    # screenshot_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/medium/item_0.png"
    # screenshot_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/bts_black_frame_1/item_4.png"
    screenshot_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/detail.jpeg"
    
    base64_image = encode_image_to_base64(screenshot_path)
    
    # input_record = {
    #     "headline": "",
    #     "publish_date": "",
    #     "number_of_views": "357",
    #     "sequence_number": "1"
    # }

    # input_record = {
    #     "book_name": "",
    #     "price": "",
    #     "stock_avail": ""
    # }

    input_record = {
        "brand": "",
        "price": "",
        "color": "",
        "manufacturer": "",
        "asin": "",
        "country_of_origin": "China",
        "date_first_avail": "November 19, 2024"
    }

    base_url = "https://router.huggingface.co/nebius/v1"
    model = "Qwen/Qwen2-VL-7B-Instruct"
    
    prompt = (
            f"You are tasked with understanding the content of {json.dumps(input_record)} and image.\n"
            "You need to reparse the record and rewrite it filling the missing values.\n"
            "Do not assume anything or hallucinate. Just fill the missing value with text extracted from image. Do not add any extra information." \
            "Try to use the **BUILT-IN OCR** capabilities of the model to extract text from the image.\n"
        )
    
    asyncio.run(fill_empty_record(
        base_url=base_url,
        model = model,
        image_url = f"data:image/jpeg;base64,{base64_image}",
        prompt = prompt, 
    ))