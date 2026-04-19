import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import google.generativeai as genai
import os
import time
import json
import re
from datetime import datetime

# ------------------ SETUP ------------------

cred = credentials.Certificate("../firebase/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

load_dotenv()
em = os.environ.get("email")

if not em:
    raise ValueError("❌ Email not found in .env file")

print(f"📧 Using email: {em}")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")


# ------------------ CLEAN FIRESTORE DATA ------------------

def convert_firestore_data(data):
    if isinstance(data, dict):
        return {k: convert_firestore_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_firestore_data(i) for i in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
        return data


# ------------------ FETCH NASA DATA ------------------

def get_nasa_data():
    doc_ref = db.collection('hackathon') \
        .document('PCCE2026') \
        .collection('NASAreport') \
        .document(f"{em}_NASAreport")

    doc = doc_ref.get()
    return doc.to_dict() if doc.exists else {}


# ------------------ EXTRACT JSON ------------------

def extract_json(text):
    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None
    return None


# ------------------ GEMINI ------------------

def generate_agri_advice(future_data, nasa_data):

    prompt = f"""
You are an expert agricultural AI.

Analyze deeply and return ONLY JSON.

DATA:
Future:
{json.dumps(future_data, indent=2)}

NASA:
{json.dumps(nasa_data, indent=2)}

FORMAT:
{{
  "summary": "",
  "crop_analysis": {{
    "is_suitable_for_farming": true,
    "confidence_score": 0-100,
    "recommended_crops": [],
    "non_recommended_crops": []
  }},
  "weather_impact": {{
    "temperature_effect": "",
    "rainfall_effect": "",
    "risk": ""
  }},
  "actions": [],
  "warnings": []
}}
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        parsed = extract_json(raw)

        if not parsed:
            print("⚠️ JSON parse failed")
            print(raw)
            return None

        return parsed

    except Exception as e:
        print("❌ Gemini Error:", e)
        return None


# ------------------ LISTENER ------------------

def listen_to_changes():

    doc_ref = db.collection('hackathon') \
        .document('PCCE2026') \
        .collection('future_pred') \
        .document(f"{em}_input")

    def on_snapshot(doc_snapshot, changes, read_time):

        for doc in doc_snapshot:
            if not doc.exists:
                print("❌ Deleted")
                return

            future_data = doc.to_dict()

            # 🚫 PREVENT LOOP: skip if AI already processed
            if future_data.get("processed") == True:
                print("⏭️ Already processed, skipping...")
                return

            print("\n🔥 New Change Detected")

            # Clean data
            clean_future = convert_firestore_data(future_data)

            # Get NASA
            nasa_data = get_nasa_data()
            clean_nasa = convert_firestore_data(nasa_data)

            print("🌍 NASA DATA:")
            print(json.dumps(clean_nasa, indent=2))

            # Gemini
            result = generate_agri_advice(clean_future, clean_nasa)

            if result:
                print("\n🤖 AI OUTPUT:")
                print(json.dumps(result, indent=2))

                # ✅ SAVE RESULT (separate collection)
                db.collection('hackathon') \
                    .document('PCCE2026') \
                    .collection('future_pred') \
                    .document(f"{em}_output") \
                    .set(result)

                # ✅ MARK AS PROCESSED (prevents loop)
                doc.reference.update({"processed": True})

                print("✅ Saved + marked processed")

            else:
                print("❌ Failed")

    doc_ref.on_snapshot(on_snapshot)
    print("👀 Listening...")


# ------------------ MAIN ------------------

if __name__ == "__main__":
    listen_to_changes()

    print("🚀 Running...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped")