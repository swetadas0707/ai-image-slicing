import requests
import os
import json

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    print(json.dumps(
        requests.get(
            "https://api.deepseek.com/v1/models", 
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}).json(), indent=4))

    # print(json.dumps(
    #     requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}).json(),
    #     indent=4))