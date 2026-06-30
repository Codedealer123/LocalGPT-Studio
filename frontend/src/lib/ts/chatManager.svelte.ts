// chatManager.svelte.ts

import { loadState, saveState } from "./chat/chatStorage";
import type {
    Chat,
    ChatState,
} from "./chat/types";
import { createChat as apiCreateChat, getChats, getMessages, deleteChat as apiDeleteChat, renameChat as apiRenameChat } from "./api";

const DEFAULT_STATE: ChatState = {
    chats: [],
    activeChatId: null,
    openChatMenuId: null
};
const ACTIVE_CHAT_KEY = "activeChatId";



function createManager() {
    const state = $state<ChatState>(
        structuredClone(DEFAULT_STATE)
    );

    let initialized = false;

    async function init() {
        try {
            const serverChats = await getChats();
    
            state.chats = serverChats.map((c: any) => ({
                id: c.id,
                title: c.title,
                createdAt: c.created_at,
                messages: []
            }));
          
            const storedActiveChatId =
                sessionStorage.getItem(ACTIVE_CHAT_KEY);
    
            if (
                storedActiveChatId &&
                state.chats.some(
                    chat => chat.id === storedActiveChatId
                )
            ) {
                state.activeChatId = storedActiveChatId;
            } else {
                state.activeChatId =
                    state.chats[0]?.id ?? null;
            }
    
        } catch (err) {
            console.error("Failed to load chats from backend:", err);
    
            try {
                const saved = await loadState();
    
                if (saved) {
                    state.chats = saved.chats;
                }
    
                const storedActiveChatId =
                    sessionStorage.getItem(ACTIVE_CHAT_KEY);
    
                if (
                    storedActiveChatId &&
                    state.chats.some(
                        chat => chat.id === storedActiveChatId
                    )
                ) {
                    state.activeChatId = storedActiveChatId;
                } else {
                    state.activeChatId =
                        state.chats[0]?.id ?? null;
                }
    
            } catch (fallbackErr) {
                console.error("Fallback load failed:", fallbackErr);
    
                state.chats = [];
                state.activeChatId = null;
            }
        }
    
        initialized = true;
    }

    init();

    let saveTimer: number;

    function syncActiveChat(id: string | null) {
        state.activeChatId = id;
    
        if (id) {
            sessionStorage.setItem(ACTIVE_CHAT_KEY, id);
        } else {
            sessionStorage.removeItem(ACTIVE_CHAT_KEY);
        }
    }

    function persist() {
        const snapshot = JSON.parse(JSON.stringify(state));
        saveState(snapshot).catch(console.error);
    }
  
    async function createChat(title = "New Chat") {
        const res = await apiCreateChat(title);
    
        const chat: Chat = {
            id: res.chat_id,
            title: res.title,
            createdAt: Date.now(),
            messages: []
        };
    
        state.chats = [chat, ...state.chats];
        syncActiveChat(chat.id);
    
        return chat;
    }

    function getChat(id: string) {
        return state.chats.find(
            c => c.id === id
        );
    }

    function getActiveChat() {
        return getChat(
            state.activeChatId ?? ""
        );
    }

    async function setActiveChat(id: string) {
        syncActiveChat(id);
    
        const chat = getChat(id);
        if (!chat) return;
    
        const res = await getMessages(id);
    
        chat.messages = res.messages;
    }

    async function renameChat(id: string, title: string) {
        await apiRenameChat(id, title);
    
        const chat = getChat(id);
        if (!chat) return;
    
        chat.title = title;
    }

    async function deleteChat(id: string) {
        await apiDeleteChat(id);
    
        state.chats = state.chats.filter(c => c.id !== id);
    
        if (state.activeChatId === id) {
            const next = state.chats[0]?.id ?? null;
            syncActiveChat(next);
        }
    }

    function createMessage(
        role: Message["role"],
        content: string
    ): Message {
        return {
            id: crypto.randomUUID(),
            role,
            content,
            createdAt: Date.now()
        };
    }

    function addMessage(
        chatId: string,
        message: Message
    ) {
        const chat = getChat(chatId);

        if (!chat) return;

        chat.messages = [
            ...chat.messages,
            message
        ];
    }

    function updateMessage(
        chatId: string,
        messageId: string,
        content: string
    ) {
        const chat = getChat(chatId);

        if (!chat) return;

        chat.messages =
            chat.messages.map(msg =>
                msg.id === messageId
                    ? {
                          ...msg,
                          content
                      }
                    : msg
            );
    }

    function removeMessage(
        chatId: string,
        messageId: string
    ) {
        const chat = getChat(chatId);

        if (!chat) return;

        chat.messages =
            chat.messages.filter(
                m => m.id !== messageId
            );
    }

    function toggleChatMenu(id: string) {
      state.openChatMenuId =
        state.openChatMenuId === id ? null : id;
    }
    
    function closeChatMenu() {
      state.openChatMenuId = null;
    }

    let offsetX = 6;
    let offsetY = -2;
    let menuPos = $state({ x: 0 + offsetX, y: 0 + offsetY });
  
    function openMenuFromRect(rect: DOMRect) {
      menuPos = {
        x: rect.right + offsetX,
        y: rect.top + offsetY
      };
    
      return menuPos;
    }

    return {
        state,
        createChat,
        getChat,
        getActiveChat,
        setActiveChat,
        renameChat,
        deleteChat,
        createMessage,
        addMessage,
        updateMessage,
        removeMessage,
        menuPos,
        openMenuFromRect,
        persist
    };
}

export const chatManager =
    createManager();