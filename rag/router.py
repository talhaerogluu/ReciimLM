import json
import re
from datetime import datetime
from services.llm_engine_with_vllm import generate_response

def analyze_and_route_query(query, history):
    """
    Geçmişe bakarak soruyu tekil hale getirir ve
    ChromaDB için metadata filtresini aynı anda üretir.
    """
    history_text = ""
    if history:
        history_text = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in history[-5:]])

    current_date = datetime.now().strftime("%Y-%m-%d")
        
    system_prompt = """
    You are an intelligent routing and contextualization engine for a hotel AI assistant.
    Your task is to analyze the user's latest query along with the chat history.
    Todays date is: {current_date}
    
    You must output ONLY a valid JSON object with EXACTLY two keys:
    1. "standalone_query": Reformulate the user's latest query so it can be understood completely without the chat history. 
       CRITICAL RULE: You MUST write the 'standalone_query' in the EXACT SAME LANGUAGE as the user's original query. Do NOT translate it to English unless the user spoke in English.
    2. "filter": Decide if the query strictly falls into specific metadata categories. Available 'type' values: "faq", "guide", "hotel_info". If general or unsure, return {}.
    3. "action": If the user is asking about ROOM AVAILABILITY, PRICES, or BOOKING for specific dates, you must output an action object.
      - If they ask for general availability without dates, try to infer dates from the query or set action to null.
      - The action format MUST be: {"name": "check_availability", "checkin": "YYYY-MM-DD", "checkout": "YYYY-MM-DD", "adults": 2, "children": 0}       - If no availability check is needed, set "action" to null.
      - IMPORTANT: Extract the number of adult guests and children. If not specified, default adults to 2 and children to 0.
      
    Example Output 1 (User speaks English):
    {
        "standalone_query": "What time does the hotel pool open?",
        "filter": {"type": "hotel_info"}
        "action": null
    }
    
    Example Output 2 (User speaks Turkish: "kahvaltı var mı?"):
    {
        "standalone_query": "Otelde kahvaltı hizmeti var mı?",
        "filter": {"type": "faq"}
        "action": null
    }

    Example Output 3 (User speaks Arabic: "هل يوجد موقف سيارات؟"):
    {
        "standalone_query": "هل يوجد موقف سيارات في الفندق؟",
        "filter": {"type": "hotel_info"}
    }

    Example Output 4 (User speaks Turkish: "15 Mayıs ile 18 Mayıs arası boş oda var mı fiyat nedir?"):
    {
        "standalone_query": "15 Mayıs ve 18 Mayıs arası 3 yetişkin 2 çocuk için boş oda ve fiyat nedir?",
        "filter": {},
        "action": {"name": "check_availability", "checkin": "2026-05-15", "checkout": "2026-05-18", "adults": 3, "children": 2}
    }
    
    No chat, no explanation, ONLY valid JSON.
    """
    
    user_prompt = f"Chat History:\n{history_text}\n\nLatest Question: {query}\n\nJSON Output:"
    
    response_text = generate_response(user_prompt, custom_system_prompt=system_prompt, max_tokens=250)
    
    # Clean output to find JSON
    json_str = response_text
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        
    try:
        parsed_json = json.loads(json_str)
        # Yeni mimaride 3 değişken döndürüyoruz
        return parsed_json.get("standalone_query", query), parsed_json.get("filter", {}), parsed_json.get("action", None)
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error in Router: {e}")
        print(f"Raw Output: {response_text}")
        return query, {}, None