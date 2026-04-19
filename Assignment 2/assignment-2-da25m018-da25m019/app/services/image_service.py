import os
import requests
from dotenv import load_dotenv

load_dotenv()

def generate_image(prompt: str) -> bytes:
    api_key = os.getenv("HF_API_KEY")

    url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print(response.status_code)
        print(response.text)
        raise Exception(response.text)

    return response.content
