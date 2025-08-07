from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
# python -m uvicorn main:app --reload
# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Or replace with ["http://localhost:3000"] or whatever your frontend URL is
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# Templates setup
templates = Jinja2Templates(directory="templates")

# Conversation history per user
conversations: Dict[str, List[Dict[str, str]]] = {}

class ChatInput(BaseModel):
    user_id: str
    message: str

@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/reset")
async def reset(input: ChatInput):
    user_id = input.user_id
    if user_id in conversations:
        del conversations[user_id]
    return {"status": "reset"}

@app.post("/chat")
async def chat(input: ChatInput):
    user_id = input.user_id
    message = input.message.strip()
    # If user sends a message, proceed normally
    if user_id not in conversations :
        conversations[user_id] = [
            {
                "role": "system",
                "content": (
                   "you were made by attila hagen. You are FitnessGPT, Act as a professional personal trainer whose role is to help users improve their physical health, fitness, and motivation. you are very exited and encouraging and motivating. you give short to the point answers. you are very friendly and helpful. "
                )
            }
        ]

    conversations[user_id].append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversations[user_id]
        )
        reply = response.choices[0].message.content
        
        conversations[user_id].append({"role": "assistant", "content": reply})
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}
