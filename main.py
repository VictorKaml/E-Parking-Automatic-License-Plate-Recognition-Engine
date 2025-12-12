import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from paddleocr import PaddleOCR

# --- INITIALIZATION ---
app = FastAPI()

# Load YOLO model once
# Ensure 'best.pt' is in your root directory
model = YOLO("best.pt")

# Initialize PaddleOCR globally
# use_gpu=False is critical for Render/CPU-only environments
PADDLE_OCR = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)

def clean_plate_result(text: str) -> str:
    """Standardizes the output text from the OCR engine."""
    return (
        text.upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("\n", "")
        .replace("O", "0") 
        .replace("I", "1")
        .replace("S", "5")
    )

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        # 1. Load and Decode Image
        contents = await image.read()
        np_array = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse({"error": "Could not decode image"}, status_code=400)

        # 2. YOLO Inference for Bounding Box
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        yolo_results = model(img_pil)[0]

        if not yolo_results.boxes:
            return JSONResponse({
                "plate_text": None, 
                "message": "No objects detected"
            })

        # 3. Filter for best plate detection
        best_box = None
        max_conf = -1.0
        
        for box in yolo_results.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls].lower()
            
            # Check if detection is a license plate
            if "License_Plate" in class_name and conf > max_conf:
                max_conf = conf
                best_box = box.xyxy[0].tolist()

        if not best_box:
            return JSONResponse({
                "plate_text": None, 
                "message": "License plate class not found"
            })

        # 4. Cropping and Padding
        x1, y1, x2, y2 = [int(v) for v in best_box]
        padding = 10
        h, w, _ = img_bgr.shape
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
        
        cropped_plate = img_bgr[y1:y2, x1:x2]

        # 5. PaddleOCR Processing
        # paddle_res structure: [ [ [box], [text, confidence] ], ... ]
        paddle_res = PADDLE_OCR.ocr(cropped_plate, cls=False)
        
        final_plate = ""
        ocr_confidence = 0.0

        if paddle_res and paddle_res[0]:
            # Extract text and internal OCR confidence
            text_parts = []
            conf_sum = 0
            for line in paddle_res[0]:
                text_parts.append(line[1][0])
                conf_sum += line[1][1]
            
            raw_text = " ".join(text_parts)
            final_plate = clean_plate_result(raw_text)
            ocr_confidence = conf_sum / len(paddle_res[0])

        return {
            "plate_text": final_plate if final_plate else None,
            "ocr_confidence": round(ocr_confidence, 4),
            "yolo_confidence": round(max_conf, 4),
            "message": "Detection successful" if final_plate else "No text found on plate"
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def health_check():
    return {"status": "ALPR Engine Online (PaddleOCR)"}
