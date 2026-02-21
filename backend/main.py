from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv
import os, json, logging

# ===== Load env =====
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="Code Intelligence API", version="3.0")
logging.basicConfig(level=logging.INFO)

# ===== CORS (env driven) =====
origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Models =====
class ReviewRequest(BaseModel):
    code: str = Field(..., min_length=1)
    focus_areas: list[str] = []

class Issue(BaseModel):
    severity: str
    description: str
    line: int | None = None

class ReviewResponse(BaseModel):
    issues: list[Issue]
    risk_score: int
    rewritten_code: str
    summary: str

# ===== Root =====
@app.get("/")
def root():
    return {"status": "running"}

# ===== Endpoint =====
@app.post("/api/review", response_model=ReviewResponse)
def review_code(req: ReviewRequest):
    if not client:
        raise HTTPException(status_code=503, detail="LLM not configured")

    focus = ", ".join(req.focus_areas) if req.focus_areas else "overall quality"

    prompt = f"""
You are a senior code reviewer.

Return ONLY valid JSON with this schema:
{{
 "issues":[{{"severity":"critical|high|medium|low","description":"string","line":number|null}}],
 "risk_score":0-100,
 "summary":"string",
 "rewritten_code":"string"
}}

Focus on: {focus}

Code:
{req.code}
"""

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        content = res.choices[0].message.content
        data = json.loads(content)

        logging.info("Review generated")

        return data

    except Exception:
        logging.exception("LLM failure")
        raise HTTPException(status_code=500, detail="AI processing error")
