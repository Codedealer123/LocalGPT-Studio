from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid

from pydantic import BaseModel

from db import (
    get_messages,
    delete_chat,
    rename_chat,
    ensure_chat,
    get_conn
)

from llmBackend.llm_service import LlamaService
from chat import run_chat_websocket


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://localgpt.cdgamez.xyz"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = LlamaService(
    weight_gb=4.5,
    quant="Q4_K_M",
    model_path="./llmBackend/testModel/qwen.gguf",
    context=4096
)

class ConnectionManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def send(self, websocket: WebSocket, data: dict):
        await websocket.send_text(json.dumps(data))


chat_manager = ConnectionManager()

class RenameChatRequest(BaseModel):
    title: str


class CreateChatRequest(BaseModel):
    title: str

@app.websocket("/chat")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    
    await chat_manager.connect(websocket)
    try:
        await run_chat_websocket(
            websocket=websocket,
            llm=llm,
            chat_manager=chat_manager
        )
    finally:
        chat_manager.disconnect(websocket)

@app.post("/chat")
def create_chat(body: CreateChatRequest):
    chat_id = str(uuid.uuid4())
    ensure_chat(chat_id, title=body.title)

    return {
        "chat_id": chat_id,
        "title": body.title
    }

@app.patch("/chat/{chat_id}")
def rename_chat_endpoint(chat_id: str, body: RenameChatRequest):
    rename_chat(chat_id, body.title)

    return {
        "status": "renamed",
        "chat_id": chat_id,
        "title": body.title
    }

@app.delete("/chat/{chat_id}")
def delete_chat_endpoint(chat_id: str):
    delete_chat(chat_id)

    return {
        "status": "deleted",
        "chat_id": chat_id
    }

@app.get("/chat/{chat_id}/messages")
def http_get_messages(chat_id: str):
    return {
        "chat_id": chat_id,
        "messages": get_messages(chat_id)
    }

@app.get("/chats")
def list_chats():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, title, created_at
        FROM chats
        ORDER BY created_at DESC
    """)

    rows = cur.fetchall()
    conn.close()

    return [dict(row) for row in rows]