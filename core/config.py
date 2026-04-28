import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Güvenlik ve Tokenlar
HF_TOKEN = os.getenv("HF_TOKEN")

# Dizin (Path) Ayarları
# Bu dosyanın bulunduğu 'core' klasörünün bir üst dizinini ana dizin kabul ediyoruz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTOR_DB_PATH = os.path.join(BASE_DIR, "chroma_db_bge_m3")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MEMORY_FILE_PATH = os.path.join(BASE_DIR, "hotel_chat_memory.json")
HASH_FILE_PATH = os.path.join(BASE_DIR, "processed_hashes.txt")
JSON_OUTPUT_PATH = os.path.join(BASE_DIR, "./datasets/iki32b.json")

# Model Ayarları
LLM_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-AWQ"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"