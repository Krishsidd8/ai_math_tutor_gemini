from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile
import uvicorn
import os, io, json, traceback, logging, base64
import numpy as np, cv2
import google.generativeai as genai
from sympy import sympify, simplify
from fastapi.middleware.cors import CORSMiddleware
import base64
import mimetypes

# -------------------- LOGGING CONFIG --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------- FASTAPI APP --------------------
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    logger.info("Received /predict request.")
    try:
        img = Image.open(file.file).convert("RGB")
        result = ocr_with_gemini(img)
        logger.info(f"Prediction result: {result}")
        return {"latex": result}
    except Exception as e:
        logger.error("Error during prediction:")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    logger.info("Received /solve request.")
    note = ""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        try:
            latex = ocr_with_gemini(img) or ""
        except Exception as e:
            logger.warning("Gemini OCR failed (possibly quota exceeded): %s", e)
            latex = ""
            note = "Gemini OCR unavailable (quota may be exceeded)."

        logger.info(f"Predicted LaTeX: {latex}")
        try:
            steps = solve_with_gemini(latex) if latex else []
        except Exception as e:
            logger.warning("Gemini solve failed (possibly quota exceeded): %s", e)
            steps = []
            if note:
                note += " "
            note += "Gemini solver unavailable (quota may be exceeded)."
        if not steps:
            logger.info("Falling back to SymPy for solution steps.")
            steps = quick_sympy_steps(latex)
            if not note:
                note = "SymPy fallback used."
        logger.info(f"Steps returned: {len(steps)}")
        response = {"latex": latex, "steps": steps}
        if note:
            response["note"] = note
        return response

    except Exception as e:
        logger.error("Error during /solve processing:")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://krishsidd8.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- GEMINI SETUP --------------------
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-pro")

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step":   {"type": "string"},
                    "detail": {"type": "string"}
                },
                "required": ["step", "detail"]
            }
        }
    },
    "required": ["steps"]
}

# -------------------- IMAGE PREPROCESSING --------------------
def preprocess_image(img: Image.Image) -> bytes:
    img = img.convert("L")  # grayscale
    img = ImageOps.autocontrast(img, cutoff=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# -------------------- OCR WITH GEMINI --------------------
def get_image_bytes_and_mime(file_path: str):
    with open(file_path, "rb") as f:
        img_bytes = f.read()
    
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type not in ["image/png", "image/jpeg"]:
        raise ValueError(f"Unsupported image format: {mime_type}")
    
    return img_bytes, mime_type

def ocr_with_gemini(img_input) -> str:
    try:
        if isinstance(img_input, str):
            with open(img_input, "rb") as f:
                img_bytes = f.read()
            mime_type, _ = mimetypes.guess_type(img_input)

        elif isinstance(img_input, bytes):
            img_bytes = img_input
            mime_type = "image/png"

        elif isinstance(img_input, Image.Image):
            buf = io.BytesIO()
            img_input.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            mime_type = "image/png"

        else:
            raise TypeError(f"Unsupported input type: {type(img_input)}")

        response = GEMINI_MODEL.generate_content(
            [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Extract the handwritten math equation(s) from this image and convert them into LaTeX format. Only return valid LaTeX code."},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(img_bytes).decode("utf-8"),
                            }
                        },
                    ],
                }
            ]
        )

        return response.text.strip() if response and response.text else ""

    except Exception as e:
        print("[ERROR] Gemini OCR failed:", e)
        return ""

# -------------------- SOLVER --------------------
def solve_with_gemini(latex_expr: str) -> list[dict]:
    logger.info("Calling Gemini for step-by-step solution...")
    prompt = (
        "You are a math tutor. Given a math expression in LaTeX, "
        "produce a clear, correct, step-by-step solution. "
        "Return only JSON matching the schema. "
        f"LaTeX: {latex_expr}"
    )
    try:
        response = GEMINI_MODEL.generate_content(
            [prompt],
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "response_schema": STEP_SCHEMA,
            },
        )
        data = json.loads(response.text)
        steps = data.get("steps", [])
        return [{"step": s.get("step", ""), "detail": s.get("detail", "")} for s in steps if isinstance(s, dict)]
    except Exception as e:
        logger.error("Gemini solve failed:")
        logger.error(traceback.format_exc())
        return [{"step": "AI solver unavailable", "detail": str(e)}]

def quick_sympy_steps(latex_expr: str) -> list[dict]:
    logger.info("Attempting fallback using SymPy...")
    try:
        if "=" in latex_expr:
            lhs_txt, rhs_txt = latex_expr.split("=", 1)
            lhs, rhs = sympify(lhs_txt), sympify(rhs_txt)
            expr = lhs - rhs
        else:
            expr = sympify(latex_expr)

        simp = simplify(expr)
        return [
            {"step": "Parse LaTeX", "detail": str(expr)},
            {"step": "Simplify",   "detail": str(simp)},
        ]
    except Exception as e:
        logger.error("SymPy simplification failed:")
        logger.error(traceback.format_exc())
        return [{"step": "Could not parse with SymPy", "detail": str(e)}]

# -------------------- START SERVER --------------------
if __name__ == "__main__":
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)