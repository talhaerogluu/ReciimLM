import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langdetect import detect
from core.config import LLM_MODEL_NAME, HF_TOKEN
from huggingface_hub import login

# HuggingFace Login (Eğer token varsa)
if HF_TOKEN:
    login(HF_TOKEN)

print("Qwen Modeli Yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)
print("✅ Qwen Modeli Hazır!")

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
    """Qwen modeline prompt gönderip cevabı alır."""
    
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

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2
    )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()