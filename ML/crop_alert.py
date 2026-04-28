import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import google.generativeai as genai
import os
import time
import json
import re
from datetime import datetime

# changes in the NASA report document.
# latest farm profile + previous AI output.
# slerts.

cred = credentials.Certificate("../firebase/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

load_dotenv()
em = os.environ.get("email")

if not em:
    raise ValueError("❌ Email not found in .env file")

print(f"📧 System Active for: {em}")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3-flash-preview")

# ------------------ DATA UTILITIES ------------------

def convert_firestore_data(data):
    if isinstance(data, dict):
        return {k: convert_firestore_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return[convert_firestore_data(i) for i in data]
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

    OUTPUT ONLY STRICT JSON EXACTLY MATCHING THE FOLLOWING SCHEMA.
    Do not add or omit fields. Pay strict attention to Data Types (Strings vs Numbers):

    {{
      "crop_alerts": {{
        "reinforcement_learning_note": "String explaining how this data validates/updates the previous strategy",
        "alert_level": "High", 
        "updated_at": "{datetime.now().isoformat()}",
        "change_summary": "String detailing environmental changes and implications",
        "simulations": {{
          "conservative": {{
            "risk": "String (e.g., Very High (potential for widespread crop failure))",
            "yield": "String (e.g., Projected 50-60% of potential yield...)",
            "profit": "String (e.g., Significant net loss)"
          }},
          "balanced": {{
            "risk": "String",
            "yield": "String",
            "profit": "String"
          }},
          "aggressive": {{
            "risk": "String",
            "yield": "String",
            "profit": "String"
          }}
        }},
        "immediate_actions":[
            "String Action 1",
            "String Action 2",
            "String Action 3"
        ]
      }},
      "financial_forecasting": {{
        "analysis": {{
            "action_plan": "String recommending business/action plan",
            "market_forecast": "String detailing market outlook",
            "financial_summary": "String detailing overall financial risk and ROI",
            "efficiency_score": 68,
            "roi_analysis": {{
                "roi_percentage": 15.91,
                "input_costs_usd": 330.0,
                "estimated_market_price_per_kg": 0.3,
                "expected_yield_kg_per_acre": 1275.0,
                "projected_profit_usd": 52.5
            }},
            "cost_saving_opportunities":[
                "String opportunity 1",
                "String opportunity 2"
            ]
        }}
      }}
    }}
    """

    try:
        # Enforcing JSON format at the model level
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text) # Since we forced JSON, we can parse directly
    except Exception as e:
        print(f"❌ Gemini Simulation Error: {e}")
        # Fallback to manual extraction if json.loads fails
        return extract_json(response.text) if response else None

# ------------------ DATA RETRIEVAL ------------------

def get_current_farm_profile():
    doc_ref = db.collection('hackathon').document('PCCE2026') \
                .collection('future_pred').document(f"{em}_input")
    doc = doc_ref.get()
    return doc.to_dict() if doc.exists else {"info": "No profile found"}

def get_previous_output():
    doc_ref = db.collection('hackathon').document('PCCE2026') \
                .collection('future_pred').document(f"{em}_output")
    doc = doc_ref.get()
    return doc.to_dict() if doc.exists else {}

# ------------------ NASA LISTENER ------------------

def listen_to_nasa_updates():
    """Trigger the AI every time the NASA report changes."""
    
    nasa_ref = db.collection('hackathon').document('PCCE2026') \
                 .collection('NASAreport').document(f"{em}_NASAreport")

    def on_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            if not doc.exists: return

            print(f"\n📡 [EVENT] New NASA Data Detected at {read_time}")
            
            nasa_data = convert_firestore_data(doc.to_dict())
            farm_profile = convert_firestore_data(get_current_farm_profile())
            last_strategy = convert_firestore_data(get_previous_output())

            print("🤖 Digital Twin recalculating based on new environment...")
            result = run_nasa_triggered_simulation(nasa_data, farm_profile, last_strategy)

            if result and "crop_alerts" in result:
                crop_data = result["crop_alerts"]
                financial_data = result["financial_forecasting"]

                # 1. Update general prediction endpoint
           

                # 2. Store in exact format requested for `crop_alerts`
                db.collection('hackathon').document('PCCE2026') \
                    .collection('crop_alerts').document(f"{em}_output") \
                    .set({
                        "trigger": "NASA_UPDATE",
                        "data": crop_data
                    })

                # 3. Store in exact format requested for `financial_forecasting`
                financial_payload = {
                    "nasa_trigger_id": f"{em}_NASAreport",
                    "timestamp": datetime.now().isoformat(),
                    "analysis": financial_data["analysis"]
                }
                db.collection('hackathon').document('PCCE2026') \
                    .collection('financial_forecasting').document(f"{em}_latest") \
                    .set(financial_payload)

                print(f"✅ Success: Alert Level [{crop_data.get('alert_level')}]")
                print(f"📝 Summary: {crop_data.get('change_summary')}")
                print(f"💰 Projected ROI: {financial_data['analysis']['roi_analysis']['roi_percentage']}%")
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