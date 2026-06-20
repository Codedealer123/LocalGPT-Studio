from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
from router import predict_intent

app = FastAPI()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

chat_manager = ConnectionManager()
commands = ConnectionManager()


async def detect_intent_async(text: str):
    return await asyncio.to_thread(predict_intent, text)

# ---- Endpoint 1: Chat ----
@app.websocket("/chat")
async def chat_ws(websocket: WebSocket):
    await chat_manager.connect(websocket)

    try:
        while True:
            raw = await websocket.receive_text()
            if raw == "ping":
                await websocket.send_text("pong")
                continue

            try:
                payload = json.loads(raw)
                text = payload.get("text")
                chat_id = payload.get("chatId")
                assistant_id = payload.get("assistantId")

                if not text:
                    continue

            except Exception:
                text = raw
                chat_id = None
                assistant_id = None

            try:
                intent_data = await detect_intent_async(text)

                response = {
                    "type": "intent",
                    "intent": intent_data.get("intent"),
                    "confidence": intent_data.get("confidence"),
                    "chatId": chat_id,
                    "assistantId": assistant_id
                }

                print("Intent: ", intent_data.get("intent"), " confidence: ", intent_data.get("confidence"))

                await chat_manager.send_personal_message(
                    json.dumps(response),
                    websocket
                )

            except Exception as e:
                error_response = {
                    "type": "error",
                    "message": str(e),
                    "chatId": chat_id,
                    "assistantId": assistant_id
                }

                await chat_manager.send_personal_message(
                    json.dumps(error_response),
                    websocket
                )

    except WebSocketDisconnect:
        chat_manager.disconnect(websocket)
