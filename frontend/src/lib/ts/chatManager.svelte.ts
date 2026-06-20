// chatManager.svelte.ts

import { loadState, saveState } from "./chat/chatStorage";
import type {
    Chat,
    ChatState,
    Message
} from "./chat/types";

const DEFAULT_STATE: ChatState = {
    chats: [],
    activeChatId: null
};

function createManager() {
    const state = $state<ChatState>(
        structuredClone(DEFAULT_STATE)
    );

    let initialized = false;

    async function init() {
        try {
            const saved =
                await loadState();

            if (saved) {
                state.chats = saved.chats;
                state.activeChatId =
                    saved.activeChatId;
            }
        } catch (err) {
            console.error(err);
        }

        initialized = true;
    }

    init();

    let saveTimer: number;

    function persist() {
        const snapshot = JSON.parse(JSON.stringify(state));
        saveState(snapshot).catch(console.error);
    }
  
    function createChat(
        title = "New Chat"
    ): Chat {
        const chat: Chat = {
            id: crypto.randomUUID(),
            title,
            createdAt: Date.now(),
            messages: []
        };

        state.chats = [
            chat,
            ...state.chats
        ];

        state.activeChatId = chat.id;

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

    function setActiveChat(
        id: string
    ) {
        state.activeChatId = id;
    }

    function renameChat(
        id: string,
        title: string
    ) {
        const chat = getChat(id);

        if (!chat) return;

        chat.title = title;
    }

    function deleteChat(id: string) {
        state.chats = state.chats.filter(
            c => c.id !== id
        );

        if (
            state.activeChatId === id
        ) {
            state.activeChatId =
                state.chats[0]?.id ??
                null;
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
        persist
    };
}

export const chatManager =
    createManager();