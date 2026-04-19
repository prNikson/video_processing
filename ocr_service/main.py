import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image
import io
import cv2


app = FastAPI(
    title="PaddleOCR API",
    description="A simple API for OCR using PaddleOCR",
    version="1.0.0"
)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='ru'
)

def ocr_pdf_paddle(image):
    print("START COMPRESSING...")
    COMPRESS_SCALE = 0.5
    width, height = image.size
    new_size = (int(width * COMPRESS_SCALE), int(height * COMPRESS_SCALE))
    img_resized = image.resize(new_size, Image.LANCZOS)
    image_array = np.array(img_resized)

    print("START PREDICTION...")
    page_text = ocr.predict(image_array)

    return page_text

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    print("IMAGE READING...")

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = ocr.predict(frame)

# Perform OCR
   # result = ocr_pdf_paddle(image)
    print("FILE PREDICTION ENDED")

    if isinstance(result, list) and len(result) > 0:
        rec_texts = result[0].get("rec_texts", [])
    elif isinstance(result, dict):
        rec_texts = result.get("rec_texts", [])
    else:
        rec_texts = []

    print("SENDING RESPONSE...")
    return JSONResponse(content=rec_texts)

