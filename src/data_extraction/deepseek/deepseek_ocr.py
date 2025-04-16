"""This script extracts data from an image using EasyOCR and then reparses the record using DeepSeek API."""


import os
import json
import asyncio
import requests
import easyocr


def extract_data_from_image(image_path: str):
    try:
        # Initializing easy ocr
        reader = easyocr.Reader(['en'], gpu=False)

        # Read the image
        results = reader.readtext(image_path)
        extracted_text = " ".join([res[1] for res in results])

        # Standardize the text to avoid misinterpretation of "??" as "2?"
        # extracted_text = extracted_text.replace('??', '?')

        print("\nExtracted Text:", extracted_text)
        return extracted_text

    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        return "No data extracted from image."

async def reparse_record(prompt: str, input_record: dict):

    system = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that can reparse records and fill in missing values.",
                }
            ],
        }
    ]
    user = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "text",
                    "text": json.dumps(input_record),
                },
            ],
        }
    ]
    params = {
        "model": "deepseek-chat",  # model must specifically support multi-modal input for images
        "messages": system + user,
        "max_tokens": 1500,
        "top_p": 0.5,
        "temperature": 0.5,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY')}",
    }
    response = requests.post(
        url="https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=params,
    )

    if response.status_code != 200:
        print(f"HTTP Error: {response.status_code}: {response.text}")
    else:
        print(response.json()["choices"][0]["message"]["content"])
        return response.json()["choices"][0]["message"]["content"]
            

if __name__=="__main__":
    image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/grid_black_frame/medium/item_0.png"
    image_data = extract_data_from_image(image_path)

    input_record = {
        "headline": "",
        "publish_date": "",
        "number_of_views": "357",
        "sequence_number": "1"
    }
    prompt = f"""
        You are tasked with understanding the content of {input_record} and {image_data}.
        You need to reparse the record and rewrite it filling the missing values.
        Do not assume anything or hallucinate. Just fill the missing value with text. Do not add any extra information.
    """
    asyncio.run(reparse_record(prompt, input_record))