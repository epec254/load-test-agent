
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from .agent import Agent

app = FastAPI()
agent = Agent()

class ChatRequest(BaseModel):
    history: List[Dict[str, Any]]
    message: str

class ChatResponse(BaseModel):
    history: List[Dict[str, Any]]
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handles a single turn of a conversation.
    """
    messages = request.history
    if not messages:
        messages = agent.get_initial_messages()

    updated_messages, response = agent.chat(messages, request.message)
    
    return ChatResponse(history=updated_messages, response=response)

@app.get("/")
async def root():
    return {"message": "Telco Agent API is running."}
