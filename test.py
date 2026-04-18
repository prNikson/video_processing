import requests
import time

def test_ocr(image_path):
    url = "http://localhost:8010/ocr"

    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    print(response.json())

test_ocr("qwen.png")
