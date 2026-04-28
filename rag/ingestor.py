import os
import json
import re
import hashlib
from pathlib import Path
from typing import List, Literal, Optional, Type
from pydantic import BaseModel, Field
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Kendi modüllerimizden içe aktarmalar
from core.config import DATASETS_DIR, HASH_FILE_PATH, JSON_OUTPUT_PATH
from services.llm_engine_with_vllm import generate_response

# --- PYDANTIC ŞEMALARI ---
class BaseMetadata(BaseModel):
    id: str = Field(description="Otomatik üretilen benzersiz ID")
    content_hash: str = Field(description="İçeriğin hash'i, benzersiz tanımlama için")
    type: str = Field(description="Veri tipi (faq, guide, hotel_info)")
    category: str = Field(description="Kategori (hotel, activities vb.)")
    title: str = Field(description="Başlık")
    tags: List[str] = Field(default_factory=list)
    source: str = Field(description="Veri kaynağı")
    lang: str = "tr"

class FAQMetadata(BaseMetadata):
    type: Literal["faq"] = "faq"
    question: str
    answer: str

class GuideMetadata(BaseMetadata):
    type: Literal["guide"] = "guide"

class HotelInfoMetadata(BaseMetadata):
    type: Literal["hotel_info"] = "hotel_info"

# --- SMART RAG INGESTOR ---
class SmartRAGIngestor:
    def __init__(self, mode: Literal["local"] = "local"):
        self.mode = mode
        self.hash_file = HASH_FILE_PATH
        
        self.model_mapping = {
            "faq": FAQMetadata,
            "guide": GuideMetadata,
            "hotel_info": HotelInfoMetadata
        }
        self.seen_hashes = set()
        self._load_hashes()

    def _generate_id(self, folder: str, filename: str, content: str):
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]
        clean_filename = Path(filename).stem.replace(" ", "_").lower()
        return f"{folder}_{clean_filename}_{content_hash}", content_hash

    def _load_hashes(self):
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.seen_hashes = set(data)
                print(f"📁 Geçmişten {len(self.seen_hashes)} adet hash yüklendi.")
            except Exception as e:
                print(f"⚠️ Hash dosyası okunamadı, sıfırdan başlanıyor: {e}")

    def _save_hashes(self):
        try:
            with open(self.hash_file, "w", encoding="utf-8") as f:
                json.dump(list(self.seen_hashes), f, indent=4)
        except Exception as e:
            print(f"⚠️ Hashler kaydedilemedi: {e}")

    def _get_local_metadata(self, content: str, target_model: Type[BaseModel]) -> dict:
        schema = target_model.model_json_schema()
        prompt = f"Analyze and return ONLY JSON based on this schema: {json.dumps(schema)}\nText: {content}"
        ingestion_prompt = "You are a data extraction assistant. Output ONLY valid JSON. No talk."
        
        response = generate_response(prompt, custom_system_prompt=ingestion_prompt)
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            return json.loads(match.group()) if match else {}
        except:
            return {}

    def _extract_metadata(self, content: str, target_model: Type[BaseModel], folder: str):
        return self._get_local_metadata(content, target_model)

    def get_splitter_for_folder(self, folder_name: str):
        if folder_name == "faq":
            return MarkdownHeaderTextSplitter(headers_to_split_on=[("###", "question_header")])
        elif folder_name == "guide":
            return MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")])
        else:
            return MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "H2"), ("###", "H3")])

    def process_directory(self, main_dir: str):
        dataset_path = Path(main_dir)
        all_enriched_chunks = []
        
        # Eğer json zaten varsa, içindeki eski verileri listeye al (üzerine yazmamak için)
        if os.path.exists(JSON_OUTPUT_PATH):
            try:
                with open(JSON_OUTPUT_PATH, "r", encoding="utf-8") as f:
                    all_enriched_chunks = json.load(f)
            except Exception as e:
                print(f"Mevcut JSON okunamadı: {e}")

        new_chunks_count = 0 

        for folder, target_model in self.model_mapping.items():
            path = dataset_path / folder
            if not path.exists(): continue

            splitter = self.get_splitter_for_folder(folder)

            for md_file in path.glob("*.md"):
                print(f"📖 İşleniyor: {folder}/{md_file.name}")
                
                with open(md_file, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                clean_text = "\n".join([line.strip() for line in raw_text.split("\n")])
                chunks = splitter.split_text(clean_text)

                for i, chunk in enumerate(chunks):
                    content = chunk.page_content
                    
                    if folder == "faq" and "question_header" in chunk.metadata:
                        content = f"Soru: {chunk.metadata['question_header']}\nCevap: {content}"

                    if len(content.strip()) < 10: continue

                    chunk_id, chunk_hash = self._generate_id(folder, md_file.name, content)
                    
                    if chunk_hash in self.seen_hashes:
                        continue 
                    
                    metadata_obj = self._extract_metadata(content, target_model, folder)
                    llm_data = metadata_obj.model_dump() if hasattr(metadata_obj, 'model_dump') else metadata_obj

                    final_meta = {
                        **llm_data,
                        "id": chunk_id,
                        "content_hash": chunk_hash,
                        "type": folder,
                        "source": md_file.name,
                        "category": llm_data.get("category", "general"),
                        "title": llm_data.get("title", md_file.stem)
                    }

                    try:
                        validated = target_model.model_validate(final_meta)
                    except:
                        validated = target_model.model_construct(**final_meta)

                    self.seen_hashes.add(chunk_hash)
                    new_chunks_count += 1
                    
                    all_enriched_chunks.append({
                        "content": content,
                        "metadata": validated.model_dump()
                    })

        if new_chunks_count > 0:
            self._save_hashes()
            
            # Güncellenmiş listeyi JSON'a yaz
            with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(all_enriched_chunks, f, ensure_ascii=False, indent=4)
                
            print(f"🚀 Toplam {new_chunks_count} YENİ chunk işlendi ve {JSON_OUTPUT_PATH} dosyasına eklendi.")
        else:
            print("⚡ Yeni bir içerik bulunamadı, mevcut veritabanı güncel.")

        return all_enriched_chunks

# Bu dosya terminalden direkt çağrıldığında ingestion işlemini başlatır
if __name__ == "__main__":
    print("-" * 50)
    print("Veritabanı Güncelleme (Ingestion) Başlıyor...")
    print("-" * 50)
    ingestor = SmartRAGIngestor(mode="local")
    ingestor.process_directory(DATASETS_DIR)
    
    # Not: Vektör veritabanını dinamik olarak burada Chroma'ya add_documents 
    # komutuyla ekleyecek ekstra bir mantık da ileride kurulabilir. 
    print("-" * 50)
    print("İşlem Tamamlandı. Lütfen sistemi (main.py) yeniden başlatın.")