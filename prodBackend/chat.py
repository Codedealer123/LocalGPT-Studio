import json,asyncio
from fastapi import WebSocket
from router import predict_intent
from routes import manageRoutes
def detect_intent_sync(text):return predict_intent(text)
async def detect_intent_async(text):return await asyncio.to_thread(detect_intent_sync,text)
async def run_chat_websocket(websocket,llm,chat_manager):
	M='confidence';G='intent';D='assistantId';C='type';B=chat_manager;A=websocket;await B.connect(A)
	try:
		while True:
			H=await A.receive_text()
			if H=='ping':await A.send_text('pong');continue
			E=json.loads(H);I=E.get('text','');J=await detect_intent_async(I);K=J[G];L=J[M];F=manageRoutes(K,L);await B.send(A,{C:G,G:K,M:L,'route':F})
			if F=='llm_chat':
				async for N in llm.stream_async(I):await B.send(A,{C:'token','content':N,D:E.get(D,'')})
				await B.send(A,{C:'done',D:E.get(D,'')})
			else:await B.send(A,{C:'error','message':f"Unknown route: {F}"})
	except Exception:B.disconnect(A);raise