from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import os
import filetype
import base64
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import json, os

app = FastAPI()
# python -m uvicorn main:app --reload
# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace with ["http://localhost:3000"] or whatever your frontend URL is
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))




# Conversation history per user
conversations: Dict[str, List[Dict[str, str]]] = {}

class ChatInput(BaseModel):
    user_id: str
    message: str

@app.get("/")
def root():
    return {"status": "ok", "service": "fitnessgpt-backend"}
@app.get("/health")
def health():
    return {"ok": True}
# store per-user profiles in memory (persist however you like)
profiles: Dict[str, Dict[str, Any]] = {}
PROFILE_DB_PATH = "profiles.json"

def load_profiles():
    global profiles
    if os.path.exists(PROFILE_DB_PATH):
        try:
            with open(PROFILE_DB_PATH, "r", encoding="utf-8") as f:
                profiles = json.load(f)
        except Exception:
            profiles = {}

def save_profiles():
    try:
        with open(PROFILE_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# Load once on startup
load_profiles()
class Profile(BaseModel):
    user_id: str
    age: Optional[int] = Field(None, ge=10, le=100)
    sex: Optional[str] = Field(None, description="male|female|other")
    height_cm: Optional[float] = Field(None, ge=80, le=260)
    weight_kg: Optional[float] = Field(None, ge=20, le=400)
    body_fat_pct: Optional[float] = Field(None, ge=2, le=70)
    activity: Optional[str] = Field(
        None, description="sedentary|light|moderate|active|athlete"
    )
    goal: Optional[str] = Field(
        "balanced", description="balanced|fat_loss|muscle_gain|recomp"
    )
    diet: Optional[str] = None
    allergies: Optional[List[str]] = None

def profile_to_context(p: Dict[str, Any]) -> str:
    """Turn a sparse profile dict into a concise system message for personalization."""
    if not p:
        return "No saved user profile. If useful, ask 1 brief follow-up to personalize."

    # Make a one-liner summary
    parts = []
    if p.get("age") is not None: parts.append(f'age {p["age"]}')
    if p.get("sex"): parts.append(p["sex"])
    if p.get("height_cm") is not None: parts.append(f'{p["height_cm"]} cm')
    if p.get("weight_kg") is not None: parts.append(f'{p["weight_kg"]} kg')
    if p.get("body_fat_pct") is not None: parts.append(f'BF {p["body_fat_pct"]}%')
    if p.get("activity"): parts.append(f'activity {p["activity"]}')
    if p.get("goal"): parts.append(f'goal {p["goal"]}')
    if p.get("diet"): parts.append(f'diet {p["diet"]}')
    if p.get("allergies"): parts.append(f'allergies {", ".join(p["allergies"])}')

    summary = " | ".join(parts)
    return (
        "User profile (use to personalize calories, macros, exercise selection, tone): "
        + (summary or "—")
        + ". If any critical field is missing for accuracy, ask exactly one clarifying question."
    )

# ---------- endpoints to save/fetch profile ----------
@app.post("/profile")
async def save_profile(p: Profile):
    # Convert to dict excluding None
    data = p.model_dump(exclude_none=True)
    uid = data.pop("user_id", None)
    if not uid:
        return {"status": "error", "msg": "Missing user_id"}
    
    profiles[uid] = data  # save in memory
    save_profiles()       # optional: persist to profiles.json
    return {"status": "saved", "profile": profiles[uid]}
@app.get("/profile")
async def get_profile(user_id: str):
    profile = profiles.get(user_id)
    if not profile:
        print(f"No profile found for user_id {user_id}")
        return {"profile": {}}
    print(f"Returning profile for user_id {user_id}: {profile}")
    return {"profile": profile}

# ---------- upgraded /chat that injects profile ----------
@app.post("/chat")
async def chat(input: ChatInput):
    user_id = input.user_id
    message = (input.message or "").strip()
    SYSTEM_PROMPT = (
        "You were made by Attila Hagen. You are FitnessGPT — an expert personal trainer, "
        "sports nutritionist, and motivational coach.\n\n"
        "Principles:\n"
        "- Be concise and practical.very Friendly, Warm, supportive tone. No fluff. Speak Gen Z slang\n"
        "- Personalize to the user's goal, schedule, equipment, preferences, and any injuries.\n"
        "- If the user feels demotivated, remind them of their goals and help them stay accountable.\n"
        "- always celebrate progress no matter how little.\n"
        "- remind the user how far he/she came already\n"
        "- Be numerate and actionable (kcal/macros, sets×reps, RPE/RIR when useful).\n"
        "- Formatting: short sections with **bold headers** and bullets; avoid walls of text.\n"
        "- End with one motivating question.\n\n"
        "Length rule: default to ~6–10 short lines; expand only if user asks for a full plan.\n"
    )

    # Initialize a bucket for this user's messages if needed
    if user_id not in conversations:
        conversations[user_id] = []

    # Build headers fresh on every call so profile stays up-to-date
    user_profile = profiles.get(user_id, {})
    header = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": profile_to_context(user_profile)},
    ]
    print(f"User {user_id} profile: {profile_to_context(user_profile)}")

    # Take the last few turns from existing history (assistant+user), discard old headers
    # Keep at most the last 16 exchanges (32 messages)
    prior = [m for m in conversations[user_id] if m["role"] in ("user", "assistant")]
    tail = prior[-32:]

    # New message
    tail.append({"role": "user", "content": message})

    # Final message list
    msgs = header + tail

    
    # Call the model
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=msgs,
            
        )
    except Exception as e:
        return {"reply": f"Error: {e}"}

    reply = response.choices[0].message.content or "I didn’t catch that. Could you rephrase?"

    # Save only user/assistant turns (headers rebuilt each time)
    conversations[user_id] = tail
    conversations[user_id].append({"role": "assistant", "content": reply})

    return {"reply": reply}
def _ext_for_bytes(b: bytes) -> str:
    # Best-effort file type (jpg/png/webp)
    kind = filetype.guess(b)
    if kind in ("jpeg", "jpg"):
        return "jpeg"
    if kind in ("png", "webp"):
        return kind
    # Fallback to jpeg (OpenAI is fine with this as a data URL)
    return "jpeg"

def _image_file_to_data_url(data: bytes) -> str:
    ext = _ext_for_bytes(data)
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"

def _openai_vision_chat(messages: List[Dict[str, Any]], model: str = "gpt-5-mini-2025-08-07") -> str:
    """
    Sends a multimodal chat request (text + image data URL) and returns assistant text.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
      
    )
    return resp.choices[0].message.content
@app.post("/analyze-meal")
async def analyze_meal(
    file: UploadFile = File(...),
    user_id: str = Form(default="anon"),
    dietary_goal: str = Form(default="balanced"),   # e.g., "fat loss", "muscle gain", "keto", "balanced"
    allergies: str = Form(default="")               # comma-separated (e.g., "peanuts, shellfish")
):
    """
    Upload a meal photo -> get rough calories, macros/micros, health score (0-10), and notes.
    """
    try:
        data = await file.read()
        if not data or len(data) == 0:
            return {"error": "Empty file."}
        if len(data) > 7_000_000:  # ~7 MB guardrail
            return {"error": "Image too large. Please upload an image under 7 MB."}

        data_url = _image_file_to_data_url(data)

        system_prompt = (
            """You are NutriVision, a careful nutrition analyst. you give short to the point answers. the titles should be in bulletpoint everything else just plain text"
            "Given a single meal photo, you MUST:\n"
            1) JSON on the first line only:
            {"items":[{"name":str,"portion_estimate":str}],"calories_kcal_range":":str","macros":{"protein_g":int,"carbs_g":int,"fat_g":int,"protein_pct":int,"carbs_pct":int,"fat_pct":int},"health_score_0_10":int
            "2) Mention 3–5 notable micronutrients (if inferable).\n"
            "3) Flag potential allergens (if visible/likely) and ultra-processed elements.\n"
            "4) Give a health score from 0–10 (0=very unhealthy, 10=very healthy) with 1–2 sentence justification.\n"
            "5) Simple tweaks to improve healthfulness aligned to the user's goal.\n"
            "Be concise, avoid medical certainty; use phrases like 'approximate' and 'looks like'.Do not add extra lines before the JSON."""
        )

        user_prompt = (
            f"my goal: {dietary_goal}. Known allergies: {allergies or 'none provided'}.\n"
            "Analyze this meal photo\n"
            
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            },
        ]

        text = _openai_vision_chat(messages)
        return {"result": text}
    except Exception as e:
        return {"error": str(e)}


# ---------- POST /rate-physique ----------

@app.post("/rate-physique")
async def rate_physique(
    file: UploadFile = File(...),
    user_id: str = Form(default="anon"),
):
    try:
        data = await file.read()
        if not data or len(data) == 0:
            return {"error": "Empty file."}
        if len(data) > 7_000_000:
            return {"error": "Image too large. Please upload an image under 7 MB."}

        data_url = _image_file_to_data_url(data)

        system_prompt = (
            """ 
            1) JSON on the first line only:
            {"bf_percent_range":"low-high","posture":{"summary":str},"muscle_balance":{"strengths":[str],"gaps":[str]},"overall_score_0_10":int}
            
            This GPT acts as a physique analysis and fitness recommendation coach. It uses uploaded full-body images and relevant user input (e.g., age, height, weight, strength levels) to evaluate body composition and give highly specific, actionable fitness feedback. It begins by presenting the most critical summary data: estimated body fat percentage, physique development level (e.g., beginner, intermediate, advanced), and a subjective visual attractiveness score from 1 to 10 (emphasizing it's purely aesthetic).It now also includes an **Aesthetic Ranking** from Iron to Supreme tier based on the user's overall aesthetic score, along with a **Warrior Archetype** designation (e.g., Spartan, Samurai) based on their physique traits. It follows with an in-depth **overall physique assessment** summarizing the user’s general conditioning, muscularity, symmetry, balance, and proportions. This overview highlights both strengths and areas for improvement.

Additional modules now include:
- **Personalized Insights**: Genetic potential assessment, strength/weakness evaluation, and sport suitability recommendations.
- **Body Composition Analysis**: BMI, somatotype classification, body fat percentage, vascularity, and muscle definition.
- **Natural Status Assessment**: A professional evaluation of whether the physique appears naturally obtained based on visual markers.

A **Muscle Group Assessment** is provided with detailed ratings (0–5 scale) for all clearly visible muscles including arms, chest, abs, back, shoulders, and legs.

**Body Ratios** are calculated, such as waist-to-hip and shoulder-to-waist, with commentary on their aesthetic impact.

A **Comprehensive Physique Analysis** follows, with a breakdown of posture, muscular imbalances, body symmetry, and skeletal alignment.

Two dedicated sections complete the package:
1. **Diet Phase** – Recommendation to bulk, cut, or maintain, based on physique. Includes calorie/macro targets, hydration, meal timing, and food examples.
2. **Training Plan** – Fully customized plan detailing split, weekly frequency, workouts, exercises, sets/reps, tempo, intensity (RPE or RIR), rest periods, form cues, and progressive overload strategy. Cardio guidance is included as needed.

The GPT finishes with a **Supplement Guide** (optional), clearly marking them as non-essential.
start with the following sentence: Here is your personalized physique review:
"""
        )

        user_prompt = (
            "analyze my physique photo."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            },
        ]

        text = _openai_vision_chat(messages)
        return {"result": text}
    except Exception as e:
        return {"error": str(e)}
