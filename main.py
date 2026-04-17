"""
CodeIQ — AI Code Review Bot
Paste code → AI reviews bugs, security, quality, improvements
Powered by Groq (free)
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ── Groq ──────────────────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    groq_client = Groq(api_key=GROQ_API_KEY)
    GROQ_AVAILABLE = True
except Exception as e:
    GROQ_AVAILABLE = False
    groq_client = None

app = FastAPI(title="CodeIQ API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

reviews: dict[str, dict] = {}
connected_clients: dict[str, list[WebSocket]] = {}

class ReviewRequest(BaseModel):
    code: str
    language: str = "python"
    context: Optional[str] = None

def call_groq(system: str, user: str, max_tokens: int = 1200) -> str:
    if not GROQ_AVAILABLE or not groq_client:
        return "AI not available — set GROQ_API_KEY"
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ── Review Agents ─────────────────────────────────────────────────────────────

async def review_bugs(code: str, language: str) -> dict:
    result = call_groq(
        f"""You are an expert {language} code reviewer specializing in bug detection. 
Find ALL bugs, logical errors, edge cases, and runtime errors. Be specific with line references.""",
        f"""Review this {language} code for bugs:

```{language}
{code}
```

List every bug found with:
- Bug description
- Why it's a problem
- How to fix it
- Severity: Critical/High/Medium/Low

If no bugs found, say so clearly.""",
        max_tokens=800
    )
    return {"category": "Bugs & Errors", "icon": "🐛", "output": result}

async def review_security(code: str, language: str) -> dict:
    result = call_groq(
        f"""You are a security expert reviewing {language} code for vulnerabilities.
Check for OWASP issues, injection risks, authentication flaws, data exposure, and insecure practices.""",
        f"""Security audit this {language} code:

```{language}
{code}
```

Check for:
- SQL/Command injection
- XSS vulnerabilities  
- Insecure data handling
- Authentication/authorization issues
- Hardcoded secrets
- Input validation gaps

Rate security risk: Critical/High/Medium/Low/Safe
Be specific about each issue found.""",
        max_tokens=800
    )
    return {"category": "Security", "icon": "🔒", "output": result}

async def review_quality(code: str, language: str) -> dict:
    result = call_groq(
        f"""You are a senior {language} engineer reviewing code quality, readability, and maintainability.
Focus on clean code principles, naming, structure, and best practices.""",
        f"""Review code quality for this {language} code:

```{language}
{code}
```

Evaluate:
- Readability and naming conventions
- Code structure and organization
- DRY principle (Don't Repeat Yourself)
- Single responsibility
- Error handling
- Comments and documentation
- Overall quality score: 1-10

Give specific, actionable feedback.""",
        max_tokens=800
    )
    return {"category": "Code Quality", "icon": "✨", "output": result}

async def review_performance(code: str, language: str) -> dict:
    result = call_groq(
        f"""You are a performance engineer analyzing {language} code for efficiency issues.
Find time complexity problems, memory leaks, unnecessary operations, and optimization opportunities.""",
        f"""Performance analysis for this {language} code:

```{language}
{code}
```

Analyze:
- Time complexity of key operations (Big O)
- Memory usage and potential leaks
- Unnecessary loops or redundant operations
- Database query efficiency (if applicable)
- Caching opportunities
- Performance score: 1-10

Give specific optimizations with expected improvement.""",
        max_tokens=800
    )
    return {"category": "Performance", "icon": "⚡", "output": result}

async def review_summary(code: str, language: str, all_reviews: list) -> dict:
    combined = "\n\n".join([f"{r['category']}:\n{r['output']}" for r in all_reviews])
    result = call_groq(
        "You are a lead engineer summarizing a complete code review.",
        f"""Summarize this complete code review for {language} code and give an overall verdict.

Reviews:
{combined}

Provide:
1. Overall Score: X/10
2. Top 3 Critical Issues to fix immediately
3. Top 3 Improvements recommended
4. What the code does well
5. One-paragraph overall verdict

Be direct and actionable.""",
        max_tokens=600
    )
    return {"category": "Summary", "icon": "📋", "output": result}

# ── Pipeline ──────────────────────────────────────────────────────────────────
async def broadcast(session_id: str, event: str, data: dict):
    dead = []
    for ws in connected_clients.get(session_id, []):
        try:
            await ws.send_json({"event": event, "data": data})
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients[session_id].remove(ws)

async def run_review_pipeline(session_id: str, code: str, language: str):
    session = reviews[session_id]
    session["status"] = "running"
    all_reviews = []

    review_fns = [
        ("Scanning for bugs...", review_bugs),
        ("Running security audit...", review_security),
        ("Checking code quality...", review_quality),
        ("Analyzing performance...", review_performance),
    ]

    for i, (msg, fn) in enumerate(review_fns):
        await broadcast(session_id, "review_start", {
            "message": msg, "step": i+1, "total": 5
        })
        result = await fn(code, language)
        all_reviews.append(result)
        session["reviews"].append(result)
        await broadcast(session_id, "review_complete", result)
        await asyncio.sleep(0.3)

    # Summary
    await broadcast(session_id, "review_start", {"message": "Generating summary...", "step": 5, "total": 5})
    summary = await review_summary(code, language, all_reviews)
    session["summary"] = summary
    session["status"] = "complete"
    await broadcast(session_id, "review_complete", summary)
    await broadcast(session_id, "pipeline_complete", {"session_id": session_id, "summary": summary["output"]})

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/api/review")
async def start_review(req: ReviewRequest):
    session_id = str(uuid.uuid4())
    reviews[session_id] = {
        "session_id": session_id,
        "language": req.language,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "reviews": [],
        "summary": None,
    }
    connected_clients[session_id] = []
    asyncio.create_task(run_review_pipeline(session_id, req.code, req.language))
    return {"session_id": session_id}

@app.get("/api/review/{session_id}")
async def get_review(session_id: str):
    return reviews.get(session_id, {"error": "Not found"})

@app.get("/api/health")
async def health():
    return {"status": "ok", "groq_available": GROQ_AVAILABLE}

@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    connected_clients.setdefault(session_id, []).append(websocket)
    try:
        await websocket.send_json({"event": "connected", "data": {"session_id": session_id}})
        while True:
            try:
                await asyncio.wait_for(websocket.receive(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"event": "ping", "data": {}})
    except WebSocketDisconnect:
        pass
    finally:
        clients = connected_clients.get(session_id, [])
        if websocket in clients:
            clients.remove(websocket)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
