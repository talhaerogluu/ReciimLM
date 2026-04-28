from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from langdetect import detect
from core.config import LLM_MODEL_NAME, HF_TOKEN
from huggingface_hub import login

# HuggingFace Login (Eğer token varsa)
if HF_TOKEN:
    login(HF_TOKEN)

print("Qwen Modeli vLLM ile Yükleniyor... (Bu işlem VRAM tahsisi yapacağı için biraz sürebilir)")

# Tokenizer'ı Qwen'in Chat Formatını (apply_chat_template) basmak için hala tutuyoruz
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

# --- KRİTİK vLLM AYARLARI ---
llm = LLM(
    model=LLM_MODEL_NAME,
    quantization="awq",           # Modelin AWQ formatında olduğunu açıkça belirtiyoruz
    gpu_memory_utilization=0.75,  # 48 * 0.75 = 36GB
    max_model_len=4096,           # Context Window'u 4K ile sınırla (OOM yememek için idealdir)
    trust_remote_code=True,       # Qwen'in özel kodunu çalıştırabilmesi için gerekli
    tensor_parallel_size=1        # Tek GPU kullanıldığını varsayıyoruz
)
print("✅ vLLM Motoru Hazır!")

def detect_language(text):
    """Metnin dilini tespit eder ve tam isim olarak döndürür."""
    try:
        lang = detect(text)
        if lang.startswith("tr"):
            return "Turkish"
        elif lang.startswith("ar"):
            return "Arabic"
        else:
            return "English"
    except:
        return "English"

def generate_response(prompt, custom_system_prompt=None):
    """vLLM motorunu kullanarak Qwen modeline prompt gönderip cevabı alır."""
    
    if custom_system_prompt:
        system_prompt = custom_system_prompt
    else:
        lang = detect_language(prompt)
        system_prompt = f"""
        You are a professional hotel receptionist.
        RULES:
        - You MUST respond ONLY in this language: {lang}
        - Do NOT switch languages.
        - Be natural, polite and helpful.
        - Speak like a real hotel receptionist (not robotic).
        """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Mesajı modelin anlayacağı raw text (ham metin) formatına dönüştürüyoruz
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Üretim Parametreleri (Eski model.generate içindeki kwargs'ın karşılığı)
    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=200,
        skip_special_tokens=True
    )

    # vLLM ile jet hızında üretim yapıyoruz
    # (use_tqdm=False diyerek terminalde gereksiz ilerleme çubuğu çıkmasını engelliyoruz)
    outputs = llm.generate([text], sampling_params=sampling_params, use_tqdm=False)
    
    # vLLM toplu (batch) işlem yapabildiği için liste döner. Biz ilk (ve tek) sonucu alıyoruz.
    response = outputs[0].outputs[0].text
    
    return response.strip()