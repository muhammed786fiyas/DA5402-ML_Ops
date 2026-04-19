import os
import requests

def translate_text(text: str, target_lang: str) -> str:
    api_key = os.getenv("TRANSLATION_API_KEY")

    url = "https://api.mymemory.translated.net/get"

    params = {
        "q": text,
        "langpair": f"en|{target_lang}"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return "Translation failed"

    data = response.json()
    return data["responseData"]["translatedText"]
