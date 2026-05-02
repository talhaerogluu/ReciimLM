from langchain_community.vectorstores import Chroma
from services.embedding_service import embedding_model
from core.config import VECTOR_DB_PATH

# ChromaDB içinde sadece önbellek için YENİ BİR KOLEKSİYON açıyoruz
cache_db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embedding_model,
    collection_name="semantic_cache_room"
)

def check_semantic_cache(user_question, threshold=0.15):
    """
    Kullanıcının sorusunu vektöre çevirip eski sorularla karşılaştırır.
    ChromaDB (L2 distance) kullanır. Mesafe (skor) ne kadar düşükse, o kadar benzerdir.
    threshold=0.15 demek -> %85 ve üzeri benzerlik varsa kabul et demektir.
    """
    # En benzeyen 1 soruyu getir
    results = cache_db.similarity_search_with_score(user_question, k=1)
    
    if results:
        doc, distance_score = results[0]
        
        # Eğer mesafe eşiğin altındaysa (yani çok benzerse)
        if distance_score < threshold:
            print(f"   CACHE DEN GELDİ! (Mesafe: {distance_score:.4f})")
            print(f"   Eşleşen Eski Soru: '{doc.page_content}'")
            return doc.metadata["ai_answer"]
            
    return None

def add_to_semantic_cache(user_question, ai_answer):
    """Qwen'in ürettiği yeni cevabı soruyla birlikte önbelleğe yazar."""
    cache_db.add_texts(
        texts=[user_question], # Vektöre çevrilecek metin (Soru)
        metadatas=[{"ai_answer": ai_answer}] # Yanında saklanacak veri (Cevap)
    )