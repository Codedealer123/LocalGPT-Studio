from PIL.ImageCms import getProfileManufacturer
from pandas.core.nanops import get_corr_func
import json
import asyncio
from fastapi import WebSocket

from router import predict_intent
from routes import manageRoutes
from db import ensure_chat, insert_message, get_messages


def detect_intent_sync(text: str):
    return predict_intent(text)


async def detect_intent_async(text: str):
    return await asyncio.to_thread(detect_intent_sync, text)


async def run_chat_websocket(websocket: WebSocket, llm, chat_manager):
    await chat_manager.connect(websocket)

    chat_id = None

    try:
        while True:
            raw = await websocket.receive_text()

            if raw == "ping":
                await chat_manager.send(websocket, "pong")
                continue

            data = json.loads(raw)

            message_text = data.get("text", "").strip()
            assistant_id = data.get("assistantId", "")

            if not message_text:
                continue

            if chat_id is None:
                chat_id = data.get("chat_id") or assistant_id or "default-chat"
                ensure_chat(chat_id=chat_id, title="Chat")

            insert_message(
                chat_id=chat_id,
                role="user",
                content=message_text
            )
            
            messages = get_messages(chat_id=chat_id, context_limit=4096)

            intent_result = await detect_intent_async(message_text)

            intent = intent_result.get("intent")
            confidence = intent_result.get("confidence")

            route = manageRoutes(intent, confidence)

            if route == "llm_chat":

                assistant_buffer = ""

                async for token in llm.stream_async(messages):
                    assistant_buffer += token

                    await chat_manager.send(websocket, {
                        "type": "token",
                        "content": token,
                        "assistantId": assistant_id or "temp"
                    })

                insert_message(
                    chat_id=chat_id,
                    role="assistant",
                    content=assistant_buffer
                )

                await chat_manager.send(websocket, {
                    "type": "done",
                    "assistantId": assistant_id or "temp"
                })

            else:
                await chat_manager.send(websocket, {
                    "type": "error",
                    "message": f"Unknown route: {route}"
                })

    except Exception:
        chat_manager.disconnect(websocket)
        raise