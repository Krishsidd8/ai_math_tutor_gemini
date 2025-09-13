from PIL import Image
from fastapi import FastAPI, File, UploadFile
import uvicorn
import os, io, json, traceback, logging
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

def ocr_with_gemini(img: Image.Image) -> str:
    logger.info("Calling Gemini for OCR Image => Latex...")
    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract the LaTeX expression from this math image. Return only the LaTeX code, no extra text."},
                    {"type": "input_image", "image_bytes": img_bytes}
                ]
            }
        ]

        response = GEMINI_MODEL.chat(messages=messages, temperature=0)
        text = response.output_text.strip()
        return text

    except Exception as e:
        logger.error("Gemini OCR failed:")
        logger.error(traceback.format_exc())
        return ""

def solve_with_gemini(latex_expr: str) -> list[dict]:
    logger.info("Calling Gemini for step-by-step solution...")
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text":
                        "You are a math tutor. Given a math expression or equation in LaTeX, "
                        "produce a clear, correct, step-by-step solution. "
                        "Only return JSON that matches the provided schema. "
                        "Avoid extra commentary. Keep steps concise but correct.\n\n"
                        f"LaTeX: {latex_expr}"}
                ]
            }
        ]

        response = GEMINI_MODEL.chat(messages=messages, temperature=0.2)
        data = json.loads(response.output_text)
        steps = data.get("steps", [])
        logger.info("Gemini returned valid response.")
        return [{"step": s.get("step", ""), "detail": s.get("detail", "")} for s in steps if isinstance(s, dict)]

    except Exception as e:
        logger.error("Gemini call failed:")
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
        logger.info("SymPy simplification successful.")
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