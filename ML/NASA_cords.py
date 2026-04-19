import requests
import google.generativeai as genai
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import traceback
import json
import re
import time

# 🔑 Replace with your Gemini API key
GEMINI_API_KEY = "AQ.Ab8RN6K5eKwbONu4z7NmuziiHIGXPGljDiycwOfpPImVHjI_wA"
import traceback

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

cred = credentials.Certificate("/Users/vedeekaparab/Desktop/GDG-DroneML/firebase/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


# ================== HELPERS ==================
def clean_values(values):
    return [v for v in values if v is not None and v not in (-999, -9999)]


def safe_avg(values, default):
    return round(sum(values) / len(values), 2) if values else default


# ================== NASA FETCH ==================
def fetch_nasa(lat, lon, date, temporal="hourly"):
    url = f"https://power.larc.nasa.gov/api/temporal/{temporal}/point"

    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": date,
        "end": date,
        "format": "JSON"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    json_data = response.json()

    if "properties" not in json_data or "parameter" not in json_data["properties"]:
        print("⚠️ Bad NASA response:", json_data)
        return None

    return json_data["properties"]["parameter"]


def process_nasa(data, date):
    temps = clean_values(list(data.get("T2M", {}).values()))
    humidity = clean_values(list(data.get("RH2M", {}).values()))
    rain = clean_values(list(data.get("PRECTOTCORR", {}).values()))
    solar = clean_values(list(data.get("ALLSKY_SFC_SW_DWN", {}).values()))

    print("Processed:", temps[:3], "...")

    if not temps:
        return None

    return {
        "date": date,
        "temp": safe_avg(temps, 25),
        "humidity": safe_avg(humidity, 60),
        "rain": round(sum(rain), 2) if rain else 0,
        "solar": safe_avg(solar, 200),
        "data_quality": {
            "temp": bool(temps),
            "humidity": bool(humidity),
            "rain": bool(rain),
            "solar": bool(solar)
        }
    }


def get_nasa_data(lat, lon):
    today = datetime.utcnow()

    # Try older days (IMPORTANT)
    for i in range(5, 12):
        target_date = today - timedelta(days=i)
        date_str = target_date.strftime('%Y%m%d')

        print(f"\n📡 Trying {date_str} (HOURLY)")

        try:
            data = fetch_nasa(lat, lon, date_str, "hourly")
            if data:
                result = process_nasa(data, target_date.strftime('%Y-%m-%d'))
                if result:
                    return result
        except Exception as e:
            print("Hourly failed:", e)

        print(f"🔁 Falling back to DAILY for {date_str}")

        try:
            data = fetch_nasa(lat, lon, date_str, "daily")
            if data:
                result = process_nasa(data, target_date.strftime('%Y-%m-%d'))
                if result:
                    return result
        except Exception as e:
            print("Daily failed:", e)

    raise ValueError("❌ No valid NASA data found")


# ================== GEMINI ==================
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def analyze_with_gemini(env_data, crop="wheat", retries=3):
    prompt = f"""
Return ONLY valid JSON.

{{
  "crop_analysis": {{
    "affect": "",
    "crop_health": "",
    "symptoms": "",
    "possible_solution": "",
    "extra_advice": ""
  }},
  "carbon_metrics": {{
    "vegetation_cover": 0,
    "residue_cover": 0,
    "soil_exposure": 0,
    "biomass_score": 0,
    "carbon_storage_tons_ha": 0
  }},
  "verification": {{
    "confidence": 0,
    "satellite_match": true
  }}
}}

Temp: {env_data['temp']} °C
Humidity: {env_data['humidity']} %
Rain: {env_data['rain']} mm
Solar: {env_data['solar']} W/m²

Analyze for {crop}.
"""

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            raw = response.text

            json_text = extract_json(raw)
            if not json_text:
                raise ValueError("No JSON found")

            return json.loads(json_text)

        except Exception as e:
            print(f"⚠️ Gemini attempt {attempt+1} failed:", e)
            time.sleep(1)

    return None

from dotenv import load_dotenv
load_dotenv()
em=os.environ.get("email")
# ================== FIREBASE ==================
def store_to_firebase(env_data, ai_data, email):
    doc_ref = db.collection("hackathon") \
        .document("PCCE2026") \
        .collection("NASAreport") \
        .document(f"{em}_NASAreport")

    doc_ref.set({
        "timestamp": datetime.utcnow(),
        "environment": env_data,
        "ai_output": ai_data,
        "email": email
    })

    print(f"✅ Stored with ID: {doc_ref.id}")


# ================== MAIN ==================
if __name__ == "__main__":
    lat, lon = 15.2993, 74.1240

    print("\n🚀 STARTING PIPELINE")

    try:
        nasa_data = get_nasa_data(lat, lon)
        print("✅ NASA:", nasa_data)

    except Exception as e:
        print("\n❌ NASA FAILED")
        print(e)
        traceback.print_exc()
        exit()

    try:
        ai_data = analyze_with_gemini(nasa_data, "wheat")
        print("✅ GEMINI:", json.dumps(ai_data, indent=2))

    except Exception as e:
        print("\n❌ GEMINI FAILED")
        print(e)
        traceback.print_exc()
        exit()

    try:
        email = em
        store_to_firebase(nasa_data, ai_data,email)

    except Exception as e:
        print("\n❌ FIREBASE FAILED")
        print(e)
        traceback.print_exc()
        exit()

    print("\n🎉 SUCCESS")