// chatController.ts
import { chatManager } from "./chatManager.svelte";
import { startWebsocket, defaultWsUri } from "./websocket";

export const websocket = startWebsocket(defaultWsUri + "chat");

let pingInterval: number | undefined;

export function initChatController() {
  websocket.addEventListener("open", () => {
    pingInterval = window.setInterval(() => {
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.send("ping");
      }
    }, 10000);
  });

  websocket.addEventListener("close", () => {
    if (pingInterval) clearInterval(pingInterval);
  });

  websocket.addEventListener("message", onMessage);
}

const buffers: Record<string, string> = {};

function onMessage(event: MessageEvent) {
  console.log("received:", event.data);
  
  let data: any;

  try {
    data = JSON.parse(event.data);
  } catch {
    return;
  }

  const chat = chatManager.getActiveChat();
  if (!chat || !data.assistantId) return;
  
  const msg = chat?.messages.find(
    m => m.id === data.assistantId
  );
  
  if (!msg) return;

  // ✅ INTENT
  if (data.type === "intent") {
    chatManager.updateMessage(
      chat.id,
      data.assistantId,
      `Intent: ${data.intent}\nConfidence: ${data.confidence}`
    );
  }

  if (data.type === "token") {
    buffers[data.assistantId] =
      (buffers[data.assistantId] || "") +
      data.content;
  
    chatManager.updateMessage(
      chat.id,
      data.assistantId,
      buffers[data.assistantId]
    );
  }

  if (data.type === "done") {
    const finalText = buffers[data.assistantId];
  
    if (finalText) {
      chatManager.updateMessage(
        chat.id,
        data.assistantId,
        finalText
      );
  
      chatManager.persist();
    }
  
    delete buffers[data.assistantId];
  }

  // ERROR
  if (data.type === "error") {
    chatManager.updateMessage(
      chat.id,
      data.assistantId,
      `Error: ${data.message}`
    );
  }
}

export function sendChatMessage(text: string) {
  console.log("sending:", text);

  
  
  let chat = chatManager.getActiveChat();
  if (!chat || chat.id == "") {
    const count =
      chatManager.state.chats.length + 1;
  
    chat = chatManager.createChat(
      `New Chat ${count}`
    );
  }

  const userMsg = chatManager.createMessage("user", text);
  chatManager.addMessage(chat.id, userMsg);

  const assistantMsg = chatManager.createMessage("assistant", "");
  chatManager.addMessage(chat.id, assistantMsg);

  websocket.send(
    JSON.stringify({
      text,
      chatId: chat.id,
      assistantId: assistantMsg.id
    })
  );
}

export function getChats() { 
  return chatManager.state.chats;
}