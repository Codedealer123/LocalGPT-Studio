from llama_cpp import Llama
import asyncio

from .llamaArgs import recommend

SYSTEM_PROMPT = "You are a helpful assistant."


class LlamaService:
    def __init__(self, weight_gb, quant, model_path, context=4096):
        config = recommend(
            weight_gb=weight_gb,
            quant=quant,
            model_path=model_path,
            context=context
        )

        self.llm = Llama(**config.to_constructor_kwargs())

    # -----------------------------
    # Sync streaming (blocking)
    # -----------------------------
    def stream_sync(self, messages: list):
        return self.llm.create_chat_completion(
            messages=messages,
            max_tokens=1536,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=True
        )

    # -----------------------------
    # Async streaming wrapper
    # -----------------------------
    async def stream_async(self, user_input: str):
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        # Runs in a background thread
        def producer():
            try:
                for chunk in self.stream_sync(user_input):
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")

                    asyncio.run_coroutine_threadsafe(
                        queue.put(content),
                        loop
                    )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        asyncio.create_task(asyncio.to_thread(producer))

        # Consumer (async generator)
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token