import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path("../storage.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_chat(chat_id: str, title: str = ""):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT OR IGNORE INTO chats (id, title, created_at)
        VALUES (?, ?, ?)
    """, (chat_id, title, int(time.time())))

    conn.commit()
    conn.close()


def insert_message(chat_id: str, role: str, content: str, message_id: str | None = None):
    conn = get_conn()
    cur = conn.cursor()

    if message_id is None:
        message_id = str(uuid.uuid4())

    cur.execute("""
        INSERT INTO messages (id, chat_id, role, content, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        message_id,
        chat_id,
        role,
        content,
        int(time.time())
    ))

    conn.commit()
    conn.close()

    return message_id

def get_messages(chat_id: str, context_limit: int = 8000):
    context_limit = context_limit - int(context_limit * 0.1)
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, chat_id, role, content, created_at
        FROM messages
        WHERE chat_id = ?
        ORDER BY created_at ASC
    """, (chat_id,))

    rows = cur.fetchall()
    conn.close()

    messages = [
        {
            "id": row["id"],
            "chat_id": row["chat_id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]

    def estimate_size(msg):
        return len(msg["content"]) + 20

    total = 0
    trimmed = []

    for msg in reversed(messages):
        size = estimate_size(msg)

        if total + size > context_limit:
            break

        trimmed.append(msg)
        total += size

    trimmed.reverse()

    return trimmed

def delete_chat(chat_id: str):
    conn = get_conn()
    cur = conn.cursor()

    # messages will auto-delete due to ON DELETE CASCADE
    cur.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    conn.commit()
    conn.close()


def rename_chat(chat_id: str, title: str):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE chats
        SET title = ?
        WHERE id = ?
    """, (title, chat_id))

    conn.commit()
    conn.close()