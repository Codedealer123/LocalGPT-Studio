// src/lib/chat/types.ts

export type MessageRole =
    | "system"
    | "user"
    | "assistant";

export type MessageStatus =
    | "sending"
    | "streaming"
    | "complete"
    | "error";

export interface ChatMessage {
    id: string;
    role: MessageRole;
    content: string;
    status: MessageStatus;
    createdAt: number;
    updatedAt: number;
}

export interface Chat {
    id: string;
    title: string;
    model: string;
    createdAt: number;
    updatedAt: number;
    messages: ChatMessage[];
}

export interface ChatState {
    version: number;
    chats: Chat[];
    activeChatId: string | null;
    openChatMenuId: string | null;
}