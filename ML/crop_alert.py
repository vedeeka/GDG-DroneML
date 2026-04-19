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

print(f"📧 System Active for: {em}")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# ------------------ DATA UTILITIES ------------------

def convert_firestore_data(data):
    if isinstance(data, dict):
        return {k: convert_firestore_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_firestore_data(i) for i in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    return data

def extract_json(text):
    text = re.sub(r"```json\n?|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except: return None
    return None

# ------------------ DATA RETRIEVAL ------------------

def get_current_farm_profile():
    """Gets the existing soil/crop data stored for the user."""
    doc_ref = db.collection('hackathon').document('PCCE2026') \
                .collection('future_pred').document(f"{em}_input")
    doc = doc_ref.get()
    return doc.to_dict() if doc.exists else {"info": "No profile found"}

def get_previous_output():
    """Gets the last strategy to see what changed."""
    doc_ref = db.collection('hackathon').document('PCCE2026') \
                .collection('future_pred').document(f"{em}_output")
    doc = doc_ref.get()
    return doc.to_dict() if doc.exists else {}

# ------------------ SIMULATION ENGINE ------------------

def run_nasa_triggered_simulation(nasa_data, farm_profile, last_strategy):
    """
    Simulates farm outcomes based on NEW NASA environmental data.
    """
    prompt = f"""
    You are an Autonomous Agricultural AI. NEW NASA DATA HAS ARRIVED.
    
    ACTION: 
    Analyze the new environmental conditions against the existing Farm Profile.
    Compare this to the 'Last Strategy' and determine if a course correction is needed.
    
    INPUTS:
    - NEW NASA Data: {json.dumps(nasa_data)}
    - Farm Profile (Soil/Crops): {json.dumps(farm_profile)}
    - Previous Strategy: {json.dumps(last_strategy)}

    OUTPUT ONLY JSON:
    {{
      "alert_level": "Low/Medium/High",
      "change_summary": "What changed in the environment and why it matters",
      "reinforcement_learning_note": "How this data improves our previous prediction",
      "simulations": {{
        "conservative": {{ "yield": "...", "profit": 0, "risk": 0 }},
        "balanced": {{ "yield": "...", "profit": 0, "risk": 0 }},
        "aggressive": {{ "yield": "...", "profit": 0, "risk": 0 }}
      }},
      "immediate_actions": [],
      "updated_at": "{datetime.now().isoformat()}"
    }}
    """

    try:
        response = model.generate_content(prompt)
        return extract_json(response.text)
    except Exception as e:
        print(f"❌ Gemini Simulation Error: {e}")
        return None

# ------------------ NASA LISTENER ------------------

def listen_to_nasa_updates():
    """Trigger the AI every time the NASA report changes."""
    
    nasa_ref = db.collection('hackathon').document('PCCE2026') \
                 .collection('NASAreport').document(f"{em}_NASAreport")

    def on_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            if not doc.exists: return

            print(f"\n📡 [EVENT] New NASA Data Detected at {read_time}")
            
            # 1. Get Context
            nasa_data = convert_firestore_data(doc.to_dict())
            farm_profile = convert_firestore_data(get_current_farm_profile())
            last_strategy = convert_firestore_data(get_previous_output())

            # 2. Run Reinforcement Learning & Simulation
            print("🤖 Digital Twin recalculating based on new environment...")
            result = run_nasa_triggered_simulation(nasa_data, farm_profile, last_strategy)

            if result:
                # 3. Store Results back to user's output
                db.collection('hackathon').document('PCCE2026') \
                    .collection('future_pred').document(f"{em}_output") \
                    .set(result)

                # 4. Store in History for Timeline/Graphing
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                db.collection('hackathon').document('PCCE2026') \
                    .collection('crop_alerts').document(f"{em}_output") \
                    .set({
                        "trigger": "NASA_UPDATE",
                        "data": result
                    })

                print(f"✅ Success: Alert Level [{result.get('alert_level')}]")
                print(f"📝 Summary: {result.get('change_summary')}")
            else:
                print("❌ Failed to process simulation.")

    nasa_ref.on_snapshot(on_snapshot)
    print(f"🛰️  Watching for NASA updates on: {em}_NASAreport")

# ------------------ MAIN ------------------

if __name__ == "__main__":
    print("🌟 NASA-Triggered Digital Twin Engine Started...")
    listen_to_nasa_updates()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down.")