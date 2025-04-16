"""This model is designed to reparse a record from an image using the BLIP model & HuggingFace Inference API."""

from huggingface_hub import InferenceClient
import json
import os
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



client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
)

# record = {
#     "product name": "In Her Wake",
#     "price": "",
#     "rating": ""
# }
record = {
        "brand": "",
        "price": "",
        "color": "",
        "manufacturer": "",
        "asin": "",
        "country_of_origin": "China",
        "date_first_avail": "November 19, 2024"
    }

prompt = (
        f"You are tasked with understanding the content of {json.dumps(record)} and image.\n"
        "You need to reparse the record and rewrite it filling the missing values.\n"
        "Do not assume anything or hallucinate. Just fill the missing value with text extracted from image. Do not add any extra information." \
        "Try to use the **BUILT-IN OCR** capabilities of the model to extract text from the image.\n"
    )

image_path = "/home/sweta/Projects/Pline/Experiment/AIScreenshotParsing/slices/detail.jpeg"
base64_image = encode_image_to_base64(image_path)

# output = client.image_to_text("item_0.png", model="Salesforce/blip-image-captioning-base")
model = "Salesforce/blip-image-captioning-base"

output = client.image_to_text(
    image=image_path,
    model=model,
)
print(output.choices[0].message)

