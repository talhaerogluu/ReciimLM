import json
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from core.config import (
    VECTOR_DB_PATH, 
    JSON_OUTPUT_PATH, 
    EMBEDDING_MODEL_NAME, 
    RERANKER_MODEL_NAME
)

print("Embedding modeli yükleniyor...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME, # BAAI/bge-m3
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True} # Vektörlerin normalize edilmesi, cosine similarity hesaplamalarında daha doğru sonuçlar verir.
)

# Sadece var olan veritabanına bağlanıyoruz (Veri ekleme işini ingestor yapacak)
vectordb = Chroma(
    persist_directory=VECTOR_DB_PATH, # "chroma_db_bge_m3"
    embedding_function=embedding_model,
    collection_name="hotel_knowledge_base"
)

# BM25 için JSON dosyasındaki güncel metinleri okuyoruz. BM25 sadece ram de çalıştığı için burada tüm dokümanları çekip hazırlıyoruz. Vektör veritabanı ise disk üzerinde çalıştığı için burada sadece embedding modeli ve yolunu tanımlamak yeterli.
documents = []
if os.path.exists(JSON_OUTPUT_PATH):
    with open(JSON_OUTPUT_PATH, "r", encoding="utf-8") as f:
        rag_data = json.load(f)
        
    for item in rag_data:
        clean_metadata = {k: v for k, v in item["metadata"].items() if not (isinstance(v, list) and len(v) == 0)}
        documents.append(Document(page_content=item["content"], metadata=clean_metadata))

# Eğer ChromaDB boşsa veya doküman sayıları eşleşmiyorsa, verileri zorla ekle
if len(documents) > 0 and vectordb._collection.count() != len(documents):
    print("⚠️ ChromaDB senkronize değil! Dokümanlar veritabanına işleniyor (Bu işlem bir kereye mahsus sürebilir)...")
    # Önce eski kayıtları temizle (Çiftlenmeyi önlemek için)
    if vectordb._collection.count() > 0:
        vectordb.delete(ids=vectordb.get()["ids"])
    
    # Yeni dokümanları ekle
    vectordb.add_documents(documents)
    print("✅ Dokümanlar ChromaDB'ye başarıyla eklendi!")

print("BM25 Motoru hazırlanıyor...")
if documents:
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 15
    print("✅ BM25 Hazır!")
else:
    bm25_retriever = None
    print("⚠️ BM25 için doküman bulunamadı. Lütfen önce veri ekleyin.")

print("Reranker modeli yükleniyor...")
reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device='cuda') # "BAAI/bge-reranker-v2-m3"
print("✅ Reranker hazır!")