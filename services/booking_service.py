import json
import requests
from pathlib import Path
from dotenv import load_dotenv
import os

# Proje kök dizinine çık (services klasöründen bir üst klasör)
BASE_DIR = Path(__file__).resolve().parent.parent

# .env dosyasının tam yolu
env_path = BASE_DIR / ".env"

# .env dosyasını yükle
load_dotenv(dotenv_path=env_path)

# DİKKAT: Bu key'i ileride .env dosyasına taşı!
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
HOTEL_ID = os.getenv("HOTEL_ID")
BASE_SERVER_URL = "https://senin-runpod-linkin-8000.proxy.runpod.net"

ROOM_MAPPING = {
    522048715: "Zemin Kat Daireleri (Daire 101)",                    # Apartment - Ground Floor
    522048712: "Orta Kat Daireleri (Daire 201 ve 202 - Arka Cephe)", # Studio with Garden View (Bahçe/Arka)
    522048709: "Orta Kat Daireleri (Daire 203 ve 204 - Ön Cephe)",   # Studio with Mountain View (Dağ/Ön)
    522048701: "Çatı Katı Daireleri (Daire 301 ve 302)"              # Apartment with Lake View (Göl/Çatı)
}

def fetch_booking_data_from_api(checkin, checkout, adults=2, children=0):
    """
    Router'dan gelen tarihleri alıp RapidAPI'ye canlı istek atar.
    Dönen sonucu direkt temizleme (parse) fonksiyonumuza gönderir.
    """
    url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/getRoomListWithAvailability"

    # API'ye göndereceğimiz parametreler (Tarihleri dinamik yaptık!)
    querystring = {
        "hotel_id": HOTEL_ID,
        "arrival_date": checkin,    # Router'dan gelecek (Örn: 2026-05-15)
        "departure_date": checkout, # Router'dan gelecek (Örn: 2026-05-18)
        "adults": str(adults),              # Standart 2 yetişkin üzerinden fiyat alıyoruz
        "room_qty": "1",
        "units": "metric",
        "temperature_unit": "c",
        "languagecode": "tr",       # Çıktıların Türkçe gelmesi için tr yaptık
        "currency_code": "EUR"      # TL bazında fiyat görmek için TRY yapabilirsin (İstersen EUR kalsın)
    }

    if children > 0:
        # Örn: children 2 ise -> "7,7" stringi oluşturur ve API'ye yollar
        querystring["children_age"] = ",".join(["7"] * children)

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "booking-com15.p.rapidapi.com"
    }

    try:
        # İstek 10 saniyeden uzun sürerse kilitlenmemesi için timeout ekledik
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        
        # Eğer sunucu hatası (404, 500) dönerse direkt except bloğuna düşürür
        response.raise_for_status() 
        
        # Karmaşık JSON'ı alıp kendi temizleme fonksiyonumuza yolluyoruz
        raw_json_data = response.json()
        return parse_booking_data(raw_json_data)
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️ RapidAPI Bağlantı Hatası: {e}")
        return "Şu anda otel sistemlerine bağlanılamıyor, lütfen daha sonra tekrar deneyin."

def parse_booking_data(data):
    """
    RapidAPI'den dönen o upuzun JSON verisini LLM'in kolayca anlayabileceği
    basit bir metne çevirir.
    """
    extracted_rooms = []
    extracted_media_urls = []
    
    # JSON yapısı "available" formatında ise
    if "available" in data:
         rooms_list = data["available"]
    # JSON yapısı "data.block" formatında ise (diğer endpoint)
    elif "data" in data and "block" in data["data"]:
        rooms_list = data["data"]["block"]
    else:
        return "İstenilen tarihler için oda bulunamadı veya veri çekilemedi."
        
    for room in rooms_list:
        # Boş (null) dönen odaları atla
        if room.get("available") is None and "available" in room:
            continue
        
        room_id = room.get("room_id")
        room_name = ROOM_MAPPING.get(room_id, room.get("name_without_policy", "Bilinmeyen Oda"))
        max_occupancy = room.get("max_occupancy", "Belirtilmemiş")
        nr_stays = room.get("nr_stays", "Belirtilmemiş")
        
        # Fiyat bilgisini çek
        price = "Fiyat Bilgisi Yok"
        price_data = room.get("product_price_breakdown", {})
        if "gross_amount" in price_data:
             price = price_data["gross_amount"].get("amount_rounded", price)
             
        refundable = "İade Edilemez" if room.get("refundable") == 0 else "İade Edilebilir"
        
        # Toparlanan veriyi metne dök
        room_info = f"- **Oda Tipi**: {room_name}\n"
        room_info += f"  - **Kapasite**: Maksimum {max_occupancy} kişi\n"
        room_info += f"  - **Konaklama Süresi**: {nr_stays} Gece\n"
        room_info += f"  - **Toplam Fiyat**: {price}\n"
        room_info += f"  - **Koşul**: {refundable}\n"
        
        if room.get("availableText"):
             room_info += f"  - **Durum**: {room.get('availableText')}\n"

        extracted_rooms.append(room_info)

        if room_id in ROOM_MAPPING:
            video_link = f"{BASE_SERVER_URL}/static/{room_id}.mp4"
            if video_link not in extracted_media_urls:
                extracted_media_urls.append(video_link)

    if not extracted_rooms:
         return "İstenilen tarihler için uygun oda bulunmuyor."

    final_text = "CANLI SİSTEM VERİSİ - ODA MÜSAİTLİĞİ VE FİYATLAR:\n"
    final_text += "\n".join(extracted_rooms)
    
    return final_text, extracted_media_urls

if __name__ == "__main__":
    # Hızlı bir test için
    with open(BASE_DIR / "getRoomList.json", "r") as f:
        sample_data = json.load(f)
        deneme = parse_booking_data(sample_data)
    with open(BASE_DIR / "yeni.txt", "w") as f_out:
        json.dump(deneme, f_out, ensure_ascii=False, indent=4)