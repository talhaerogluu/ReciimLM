from core.memory import memory_manager
from services.llm_engine_with_vllm import detect_language, generate_response
from rag.router import analyze_and_route_query
from rag.retriever import get_retrieved_context

def rag_ask(user_question, session_id):
    """Hafızalı ve Hızlandırılmış RAG Ana Fonksiyonu"""
    
    # 1. Hafızayı Çek
    chat_history = memory_manager.get_history(session_id)
    
    # 2. Soruyu Analiz Et ve Yönlendir
    search_query, query_filter = analyze_and_route_query(user_question, chat_history)
    print(f"🔍 Aranan Sorgu: {search_query}")
    
    # 3. Bağlamı Çek (Reranker & Hibrit Arama)
    context = get_retrieved_context(search_query, query_filter)
    
    # 4. Dil Tespiti ve Prompt Hazırlığı
    lang = detect_language(user_question)
    
    system_prompt = f"""
    You are a professional, helpful hotel receptionist and local guide.

    RULES:
    - You MUST respond ONLY in this language: {lang}
    - Do NOT switch languages.
    - Do NOT include any other language.
    - Be natural, polite and helpful.
    - Do NOT use unnatural or technical phrases.
    - You will be provided with CONTEXT information extracted from a reliable database.
    - Base your answer EXCLUSIVELY on the provided CONTEXT. 
    - Speak like a real human receptionist, not a robotic system.

    BEHAVIOR RULES:
    - If the user is asking about booking, availability, or pricing:
        → Ask for missing information (date, number of guests, room type)
    - If the user is asking general questions (travel, location, activities, food, etc.):
        → Do NOT ask booking-related questions
        → Just provide helpful and relevant information
    - If the question is unclear:
        → Ask a clarifying question
    """

    history_text = ""
    if chat_history:
        history_text = "PREVIOUS CHAT HISTORY:\n"
        for msg in chat_history[-3:]: 
            history_text += f"User: {msg['user']}\nAssistant: {msg['ai']}\n"
        history_text += "---------------------\n"

    user_prompt = f"""
    {history_text}
    CONTEXT INFORMATION:
    ---------------------
    {context}
    ---------------------

    USER QUESTION: 
    {user_question}

    INSTRUCTIONS:
    1. Given the context information above, answer the user's question.
    2. If the answer is not contained within the CONTEXT, politely apologize in {lang} and state that you do not have that information. DO NOT make up facts.
    3. CRITICAL: Your entire final response MUST be written perfectly in {lang}.
    """

    # 5. Nihai Cevabı Üret
    ai_response = generate_response(user_prompt, custom_system_prompt=system_prompt)
    
    # 6. Hafızaya Kaydet
    memory_manager.add_message(session_id, user_question, ai_response)
    
    return ai_response

# --- TEST SENARYOSU ---
if __name__ == "__main__":
    kullanici_id = "test_user_001"
    
    print("-" * 50)
    soru1 = "Otelin giriş ve çıkış saatleri nedir?"
    cevap1 = rag_ask(soru1, kullanici_id)
    print(f"\nSoru: {soru1}\nCevap: {cevap1}\n")
    
    print("-" * 50)
    soru2 = "Peki bu saatleri esnetme şansımız var mı?"
    cevap2 = rag_ask(soru2, kullanici_id)
    print(f"\nSoru: {soru2}\nCevap: {cevap2}\n")