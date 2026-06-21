from fastapi import FastAPI,WebSocket
import json
from llmBackend.llm_service import LlamaService
from chat import run_chat_websocket
app=FastAPI()
llm=LlamaService(weight_gb=4.5,quant='Q4_K_M',model_path='./llmBackend/testModel/qwen.gguf',context=4096)
class ConnectionManager:
	def __init__(A):A.active_connections=set()
	async def connect(B,websocket):A=websocket;await A.accept();B.active_connections.add(A)
	def disconnect(A,websocket):A.active_connections.discard(websocket)
	async def send(A,websocket,data):await websocket.send_text(json.dumps(data))
chat_manager=ConnectionManager()
@app.websocket('/chat')
async def chat_ws(websocket):await run_chat_websocket(websocket=websocket,llm=llm,chat_manager=chat_manager)