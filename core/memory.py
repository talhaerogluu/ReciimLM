import json
import os
from datetime import datetime, timedelta
from core.config import MEMORY_FILE_PATH

class MemoryManager:
    def __init__(self, file_path=MEMORY_FILE_PATH, expiry_days=3):
        self.file_path = file_path
        self.expiry_days = expiry_days
        self.memory = self._load_and_cleanup()

    def _load_and_cleanup(self):
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            now = datetime.now()
            cleaned_data = {}
            deleted_count = 0
            
            for user_id, session_data in data.items():
                last_updated = datetime.fromisoformat(session_data["last_updated"])
                if now - last_updated <= timedelta(days=self.expiry_days):
                    cleaned_data[user_id] = session_data
                else:
                    deleted_count += 1
            
            if deleted_count > 0:
                print(f"🧹 Süresi dolmuş {deleted_count} oturum temizlendi.")
            return cleaned_data
        except Exception as e:
            print(f"⚠️ Hafıza dosyası okunamadı, sıfırlanıyor: {e}")
            return {}

    def _save_memory(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"⚠️ Hafıza dosyaya yazılamadı: {e}")

    def get_history(self, user_id):
        if user_id in self.memory:
            return self.memory[user_id]["history"]
        return []

    def add_message(self, user_id, user_msg, ai_msg):
        history = self.get_history(user_id)
        history.append({"user": user_msg, "ai": ai_msg})
        
        self.memory[user_id] = {
            "last_updated": datetime.now().isoformat(),
            "history": history
        }
        self._save_memory()

# Yöneticimizi global olarak başlatıp dışarıya açıyoruz
memory_manager = MemoryManager()