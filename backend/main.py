from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import os
import re

load_dotenv()
app = FastAPI()

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ===== Models =====
class CodeReviewRequest(BaseModel):
    code: str
    focus_areas: list[str] = []

class CodeReviewResponse(BaseModel):
    analysis: str
    rewritten_code: str
    metrics: dict

# ===== Root =====
@app.get("/")
def root():
    return {"status": "API running"}

# ===== Serve Frontend =====
@app.get("/app", response_class=HTMLResponse)
def serve_app():
    html_file = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    return HTMLResponse(html_file.read_text(encoding="utf-8"))

# ===== Parser =====
def parse_review_response(review_text: str) -> dict:
    critical = re.search(r"Critical Issues:(.*?)(High Priority:|$)", review_text, re.DOTALL)
    high = re.search(r"High Priority:(.*?)(Medium Priority:|$)", review_text, re.DOTALL)
    medium = re.search(r"Medium Priority:(.*?)(Low Priority:|$)", review_text, re.DOTALL)
    low = re.search(r"Low Priority:(.*)", review_text, re.DOTALL)

    def count(section):
        return len([l for l in section.group(1).split("\n") if l.strip()]) if section else 0

    return {
        "critical_count": count(critical),
        "high_count": count(high),
        "medium_count": count(medium),
        "low_count": count(low),
    }

# ===== Review Endpoint =====
@app.post("/api/review", response_model=CodeReviewResponse)
def review_code(request: CodeReviewRequest):
    try:
        if not request.code.strip():
            raise HTTPException(status_code=400, detail="Code cannot be empty")

        focus_str = ", ".join(request.focus_areas) if request.focus_areas else "general quality"

        # ---------- ANALYSIS PROMPT ----------
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

        analysis_text = analysis_res.choices[0].message.content

        # ---------- REWRITE PROMPT ----------
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

        rewritten_code = rewrite_res.choices[0].message.content.strip()
        metrics = parse_review_response(analysis_text)

        return CodeReviewResponse(
            analysis=analysis_text,
            rewritten_code=rewritten_code,
            metrics=metrics
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))