"""This script is designed to extract data from an image using EasyOCR and then reparse the record using Grepsr's hosted endpoint + Qwen 2.5 7B chat model."""


from openai import OpenAI
import asyncio
import json
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


async def fill_empty_record(
        base_url: str,
        model: str,
        input_record: dict,
        prompt: str, 
    ):

    client = OpenAI(
        base_url=base_url,
        api_key="ollama"
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
                        "type": "text",
                        "text": json.dumps(input_record),
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
    image_data = extract_data_from_image(screenshot_path)
    
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

    base_url = "https://ml.int.grepsr.net/v1"
    model = "qwen2.5:7b"
    
    prompt = (
            f"You are tasked with understanding the content of {json.dumps(input_record)} and {image_data}.\n"
            "You need to reparse the record and rewrite it filling the missing values.\n"
            "Do not assume anything or hallucinate. Just fill the missing value with text extracted from image. Do not add any extra information." \
            "Try to use the **BUILT-IN OCR** capabilities of the model to extract text from the image.\n"
        )
    
    asyncio.run(fill_empty_record(
        base_url=base_url,
        model = model,
        input_record=input_record,
        prompt = prompt, 
    ))