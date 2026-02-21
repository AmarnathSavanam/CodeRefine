from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import os
import re
import logging

# ===== Load Environment =====
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=API_KEY)

# ===== App Init =====
app = FastAPI(title="AI Code Review API", version="2.0")

logging.basicConfig(level=logging.INFO)

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Models =====
class CodeReviewRequest(BaseModel):
    code: str = Field(..., min_length=1)
    focus_areas: list[str] = Field(default_factory=list)

class CodeReviewResponse(BaseModel):
    analysis: str
    rewritten_code: str
    metrics: dict

# ===== Root =====
@app.get("/")
def root():
    return {"status": "API running", "service": "AI Code Review"}

# ===== Serve Frontend =====
@app.get("/app", response_class=HTMLResponse)
def serve_app():
    html_file = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if not html_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(html_file.read_text(encoding="utf-8"))

# ===== Metrics Parser =====
def parse_review_response(review_text: str) -> dict:
    sections = {
        "critical": r"Critical Issues:(.*?)(High Priority:|$)",
        "high": r"High Priority:(.*?)(Medium Priority:|$)",
        "medium": r"Medium Priority:(.*?)(Low Priority:|$)",
        "low": r"Low Priority:(.*)"
    }

    def count(pattern):
        match = re.search(pattern, review_text, re.DOTALL)
        if not match:
            return 0
        return len([l for l in match.group(1).split("\n") if l.strip()])

    return {
        "critical_count": count(sections["critical"]),
        "high_count": count(sections["high"]),
        "medium_count": count(sections["medium"]),
        "low_count": count(sections["low"]),
    }

# ===== Review Endpoint =====
@app.post("/api/review", response_model=CodeReviewResponse)
def review_code(request: CodeReviewRequest):
    try:
        focus_str = ", ".join(request.focus_areas) if request.focus_areas else "general quality"

        analysis_prompt = f"""
You are a senior code reviewer.

Analyze the following code focusing on {focus_str}.

Return findings strictly in this format:

Critical Issues:
High Priority:
Medium Priority:
Low Priority:

Code:
{request.code}
"""

        analysis_res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1500,
            messages=[{"role": "user", "content": analysis_prompt}],
        )

        analysis_text = analysis_res.choices[0].message.content or ""

        rewrite_prompt = f"""
Rewrite the following code fixing bugs, improving readability,
and following best practices. Return ONLY code.

Code:
{request.code}
"""

        rewrite_res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1500,
            messages=[{"role": "user", "content": rewrite_prompt}],
        )

        rewritten_code = (rewrite_res.choices[0].message.content or "").strip()
        metrics = parse_review_response(analysis_text)

        logging.info("Review completed successfully")

        return CodeReviewResponse(
            analysis=analysis_text,
            rewritten_code=rewritten_code,
            metrics=metrics
        )

    except Exception as e:
        logging.exception("Review failed")
        raise HTTPException(status_code=500, detail="Internal AI processing error")
