import base64
import requests
import os
import json
import asyncio
import logging as log


async def find_fields(ss_path: str, records: str, retries: int = 3):
    """Find fields in the screenshot."""
    attempt = 0
    while attempt < retries:
        try:
            # Load the image an encode it
            with open(ss_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            prompt = f"""
            Analyze the given image of a webpage and fill the empty records {records} with values from the image. Use the sequence number (`sequence_number`)
            to identify the correct listed item in the image before filling the record, the sequence number starts from 1. The records are in the JSON format, return it as it is with
            the addition of value instead of empty string.
            """

            # A system message must indicate vision ability, or face denials
            system = [
                {
                    "role": "system",
                    "content": """
                You are ChatPal, an AI assistant powered by GPT-4o, with computer vision.

                Built-in vision capabilities:
                - extract text from image
                - describe images
                - analyze image contents
                """.strip(),
                }
            ]

            # A user message "content" is now an array of type objects instead of a string
            user = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                }
            ]

            # Construct the dict. as a streamable input to the JSON parameter
            params = {
                "model": "gpt-4o",  # model must specifically support multi-modal input for images
                "messages": system + user,
                "max_tokens": 1500,
                "top_p": 0.5,
                "temperature": 0.5,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            }
            response = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=params,
            )
            if response.status_code != 200:
                print(f"HTTP Error: {response.status_code}: {response.text}")
            else:
                print(response.json()["choices"][0]["message"]["content"])
                return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            log.error(f"Error on attempt {attempt + 1} while finding fields: {e}")
            attempt += 1
            if attempt < retries:
                await asyncio.sleep(5)
            else:
                log.error("Max retries reached. Returning null...")
                return None
            

async def main():
    ss_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/book_items/row_2.png"
    with open("/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/src/records.json", "r") as file:
        records = json.load(file)
    response = await find_fields(ss_path, records)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())