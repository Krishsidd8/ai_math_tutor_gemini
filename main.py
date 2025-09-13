from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile
import uvicorn
import os, io, json, traceback, logging, base64
import numpy as np, cv2
import google.generativeai as genai
from sympy import sympify, simplify
from fastapi.middleware.cors import CORSMiddleware

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
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        latex = ocr_with_gemini(img) or ""
        logger.info(f"Predicted LaTeX: {latex}")

        steps = solve_with_gemini(latex)
        if not steps:
            logger.warning("Gemini returned no steps. Falling back to SymPy.")
            steps = quick_sympy_steps(latex)

        logger.info(f"Steps returned: {len(steps)}")
        return {"latex": latex, "steps": steps}
    except Exception as e:
        logger.error("Error during solve:")
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
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)
    arr = np.array(img)
    arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    arr = cv2.medianBlur(arr, 3)
    ok, buf = cv2.imencode(".png", arr)
    return buf.tobytes() if ok else None

# -------------------- OCR WITH GEMINI --------------------
def ocr_with_gemini(img: Image.Image) -> str:
    logger.info("Calling Gemini for OCR Image => Latex...")
    try:
        img_bytes = preprocess_image(img)
        if not img_bytes:
            raise RuntimeError("Preprocessing failed")

        SYSTEM_PROMPT = """
        You are an OCR-to-LaTeX assistant. Extract all handwritten math equations.
        Respond only in JSON with this schema:
        {
          "equations": ["<eqn1 LaTeX>", "<eqn2 LaTeX>", ...]
        }
        No commentary, no markdown fences, no extra text.
        """

        response = GEMINI_MODEL.generate_content(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Here is the math image.", "mime_type": "image/png", "data": img_bytes},
            ],
            generation_config={
                "temperature": 0,
                "max_output_tokens": 512,
                "response_mime_type": "application/json",
            }
        )

        # Try parsing JSON
        try:
            data = json.loads(response.text)
            eqns = data.get("equations", [])
            return " ".join(eqns).strip()
        except Exception:
            logger.warning("Response not JSON, returning raw text")
            return response.text.strip()

    except Exception as e:
        logger.error("Gemini OCR failed:")
        logger.error(traceback.format_exc())
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