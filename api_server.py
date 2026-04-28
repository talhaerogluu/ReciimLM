## İLERİNİN API SUNUCUSU: POWERED BU FASTAPI


from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn

# Yazdığımız o devasa mimariyi tek satırda içeri alıyoruz!
from main import rag_ask 

app = FastAPI(title="Anemo Suit WhatsApp AI Backend")

# --- VERİ MODELLERİ ---
class ChatRequest(BaseModel):
    user_id: str
    message: str

# 1. HEALTH CHECK
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Otel AI Backend'i Jet Gibi Çalışıyor! 🚀"}

# 2. LOKAL TEST ENDPOINT'İ (Postman için)
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    print(f"[{req.user_id}] Soruyor: {req.message}")
    
    # RAG Motorumuzu tetikliyoruz
    cevap = rag_ask(req.message, req.user_id)
    
    return {"response": cevap}

# 3. WHATSAPP WEBHOOK DOĞRULAMA (Meta'nın İlk Testi)
@app.get("/webhook")
def verify_webhook(request: Request):
    # Bu token'ı Meta panelinde biz belirleyeceğiz, şimdilik böyle kalsın
    VERIFY_TOKEN = "AZEYİP GİZLİ TOKEN" 
    
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("✅ Webhook Meta tarafından doğrulandı!")
            # Meta, 'challenge' değerini integer olarak geri dönmemizi ister
            return int(challenge) 
        else:
            raise HTTPException(status_code=403, detail="Token uyuşmazlığı!")
            
    raise HTTPException(status_code=400, detail="Geçersiz istek")

# 4. WHATSAPP'TAN GELEN MESAJLARI KARŞILAMA
@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    # Meta'dan gelen devasa JSON verisini yakalıyoruz
    payload = await request.json()
    
    try:
        # WhatsApp JSON'ının içi çok iç içedir, mesajı ve numarayı cımbızla çekeriz
        entry = payload.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])
        
        if messages:
            msg = messages[0]
            telefon_no = msg.get("from") # Kullanıcının telefon numarası (session_id olacak)
            gelen_mesaj = msg.get("text", {}).get("body")
            
            print(f"📱 WhatsApp'tan Mesaj Var! | Numara: {telefon_no} | Mesaj: {gelen_mesaj}")
            
            # --- RAG MOTORU DEVREYE GİRİYOR ---
            # cevap = rag_ask(gelen_mesaj, telefon_no)
            
            # TODO: Üretilen bu cevabı Meta API'si üzerinden tekrar WhatsApp'a geri gönderecek kodu yazacağız.
            
    except Exception as e:
        print(f"⚠️ Webhook işlenirken hata: {e}")

    # Meta'ya "Mesajı aldım, sorun yok" demek için 200 dönmek ZORUNDAYIZ, yoksa bizi spamlar.
    return {"status": "success"}

if __name__ == "__main__":
    # Sunucuyu 8000 portunda başlatıyoruz
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)