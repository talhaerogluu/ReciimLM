import json
import re
from services.llm_engine_with_vllm import generate_response

def analyze_and_route_query(query, history):
    """
    Geçmişe bakarak soruyu tekil hale getirir ve
    ChromaDB için metadata filtresini aynı anda üretir.
    """
    history_text = ""
    if history:
        history_text = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in history[-5:]])
        
    system_prompt = """
    You are an intelligent routing and contextualization engine for a hotel AI assistant.
    Your task is to analyze the user's latest query along with the chat history.
    
    You must output ONLY a valid JSON object with EXACTLY two keys:
    1. "standalone_query": Reformulate the user's latest query so it can be understood completely without the chat history. 
       CRITICAL RULE: You MUST write the 'standalone_query' in the EXACT SAME LANGUAGE as the user's original query. Do NOT translate it to English unless the user spoke in English.
    2. "filter": Decide if the query strictly falls into specific metadata categories. Available 'type' values: "faq", "guide", "hotel_info". If general or unsure, return {}.
    
    Example Output 1 (User speaks English):
    {
        "standalone_query": "What time does the hotel pool open?",
        "filter": {"type": "hotel_info"}
    }
    
    Example Output 2 (User speaks Turkish: "kahvaltı var mı?"):
    {
        "standalone_query": "Otelde kahvaltı hizmeti var mı?",
        "filter": {"type": "faq"}
    }

    Example Output 3 (User speaks Arabic: "هل يوجد موقف سيارات؟"):
    {
        "standalone_query": "هل يوجد موقف سيارات في الفندق؟",
        "filter": {"type": "hotel_info"}
    }
    
    No chat, no explanation, ONLY valid JSON.
    """
    
    user_prompt = f"Chat History:\n{history_text}\n\nLatest Question: {query}\n\nJSON Output:"
    
    response = generate_response(user_prompt, custom_system_prompt=system_prompt)
    
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return result.get("standalone_query", query), result.get("filter", {})
    except Exception as e:
        print("⚠️ Router JSON parse hatası:", e)
        
    return query, {}