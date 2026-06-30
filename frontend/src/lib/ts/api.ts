const BASE_URL = "http://localhost:8000";

export async function createChat(title: string) {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title })
  });

  return res.json();
}

export async function getChats() {
  const res = await fetch(`${BASE_URL}/chats`);
  return res.json();
}

export async function getMessages(chatId: string) {
  const res = await fetch(`${BASE_URL}/chat/${chatId}/messages`);
  return res.json();
}

export async function renameChat(chatId: string, title: string) {
  return fetch(`${BASE_URL}/chat/${chatId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title })
  });
}

export async function deleteChat(chatId: string) {
  return fetch(`${BASE_URL}/chat/${chatId}`, {
    method: "DELETE"
  });
}