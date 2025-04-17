"""This script is designed to test multiple google models for data extraction from image.
    Models:
        - Gemini 1.5 Pro ✅
        - Gemini 1.5 Flash ✅
        - Gemini 1.5 Flash-8 B ✅
        - Gemini 2.0 Flash ✅
        - Gemini 2.0 Flash-Lite ✅
        - Gemini 2.5 Pro Preview 03-25 ❌
"""

import os
import time
import asyncio
import base64
import json
from google import genai
from google.genai import types
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


async def test_on_image_path(client, model, image_path, prompt):
    img_file = client.files.upload(
        file=image_path
    )
    
    response = client.models.generate_content(
        model=model,
        contents=[img_file, prompt]
    )

    print(response.text)


async def test_on_base64_image(client, model, image_path, prompt):
    img_bytes = encode_image_to_base64(image_path=image_path)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=img_bytes,
                mime_type='image/jpeg'
            ),
            prompt
        ]
    )

    print(response.text)


if __name__=="__main__":

    # image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/detail.jpeg"
    # record = {
    #         "brand": "",
    #         "price": "",
    #         "color": "",
    #         "manufacturer": "",
    #         "asin": "",
    #         "country_of_origin": "China",
    #         "date_first_avail": "November 19, 2024"
    #     }

    # image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/medium/item_0.png"
    # record = {
    #     "headline": "",
    #     "publish_date": "",
    #     "number_of_views": "357",
    #     "sequence_number": "1"
    # }
    
    image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/bts_black_frame_1/item_4.png"
    record = {
        "book_name": "",
        "price": "",
        "stock_avail": ""
    }


    prompt = (
            f"You are tasked with understanding the content of {json.dumps(record)} and image.\n"
            "You need to reparse the record and rewrite it filling the missing values.\n"
            "Do not assume anything or hallucinate. Just fill the missing value with text extracted from image. Do not add any extra information." \
            "Try to use the **BUILT-IN OCR** capabilities of the model to extract text from the image.\n"
        )
    
    model = "gemini-2.5-pro-preview-03-25"
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    t1 = time.time()
    asyncio.run(test_on_image_path(client, model, image_path, prompt))
    print("Time taken for direct file upload: ", time.time() - t1)

    t2 = time.time()
    asyncio.run(test_on_base64_image(client, model, image_path, prompt))
    print("Time taken for base64 image upload: ", time.time() - t2)