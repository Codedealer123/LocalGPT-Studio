// src/lib/chat/chatStorage.ts

import type { ChatState } from "./chatTypes";

const DB_NAME = "chat-db";
const STORE_NAME = "app-state";
const STATE_KEY = "chat-state";

async function openDB(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, 1);

        request.onupgradeneeded = () => {
            const db = request.result;

            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME);
            }
        };

        request.onsuccess = () => resolve(request.result);

        request.onerror = () => reject(request.error);
    });
}

export async function saveState(
    state: ChatState
): Promise<void> {
    const db = await openDB();

    return new Promise((resolve, reject) => {
        const tx = db.transaction(
            STORE_NAME,
            "readwrite"
        );

        const store = tx.objectStore(STORE_NAME);

        store.put(state, STATE_KEY);

        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

export async function loadState(): Promise<ChatState | null> {
    const db = await openDB();

    return new Promise((resolve, reject) => {
        const tx = db.transaction(
            STORE_NAME,
            "readonly"
        );

        const store = tx.objectStore(STORE_NAME);

        const request = store.get(STATE_KEY);

        request.onsuccess = () => {
            resolve(request.result ?? null);
        };

        request.onerror = () => {
            reject(request.error);
        };
    });
}