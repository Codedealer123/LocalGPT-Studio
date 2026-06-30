def manageRoutes(intent, confidence):
    C = 'llm_chat'
    B = 'chat_with_llm'
    A = intent
    if confidence > 0.5:
        A = B
    D = {B: C}
    return D.get(A, C)
