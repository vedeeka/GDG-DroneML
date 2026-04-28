import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import google.generativeai as genai
import os
import time
import json
from datetime import datetime

# Any NASA update → triggers simulation

cred = credentials.Certificate("../firebase/serviceAccountKey.json")
# Prevent re-initialization if script restarts in some environments
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

load_dotenv()
em = os.environ.get("email")

if not em:
    raise ValueError("❌ Email not found in .env")

print(f"📧 System Active for: {em}")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3-flash-preview")

# ------------------ HELPER: JSON CLEANER ------------------

def make_json_serializable(data):
    """Convert Firestore timestamps & complex objects to JSON-safe format"""
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(i) for i in data]
    elif hasattr(data, "isoformat"):  # handles DatetimeWithNanoseconds
        return data.isoformat()
    else:
        return data

# ------------------ CORE AI ENGINE ------------------

def run_financial_and_crop_optimizer(nasa_data, farm_profile):
    """
    Core feature: Analyzes both the financial ROI and the physical crop/weather impact
    """

    nasa_data_clean = make_json_serializable(nasa_data)
    farm_profile_clean = make_json_serializable(farm_profile)

    prompt = f"""
    You are an Expert Agricultural & Financial AI.
    Analyze the NEW NASA Weather Data against the Farm Profile.
    
    NASA DATA (Current Environment): {json.dumps(nasa_data_clean)}
    FARM PROFILE (Current Target Crop/Soil): {json.dumps(farm_profile_clean)}

    OUTPUT ONLY STRICT JSON EXACTLY MATCHING THE FOLLOWING SCHEMA.
    Pay strict attention to Data Types (Numbers vs Strings vs Booleans). Do not omit or add extra keys.

    {{
      "financial_analysis": {{
        "action_plan": "String recommending specific business/action plan",
        "roi_analysis": {{
          "roi_percentage": 15.91,
          "input_costs_usd": 330.0,
          "estimated_market_price_per_kg": 0.3,
          "expected_yield_kg_per_acre": 1275.0,
          "projected_profit_usd": 52.5
        }},
        "market_forecast": "String detailing market outlook (e.g., Neutral to Slightly Bearish)",
        "efficiency_score": 68,
        "financial_summary": "String explaining the financial situation and ROI implications",
        "cost_saving_opportunities": [
          "String opportunity 1",
          "String opportunity 2"
        ]
      }},
      "future_pred_output": {{
        "weather_impact": {{
          "risk": "String detailing the primary risks based on temperature and humidity",
          "temperature_effect": "String detailing how current temp affects the crop",
          "rainfall_effect": "String detailing how humidity/rain affects the crop"
        }},
        "summary": "String providing an overall crop viability summary based on the conditions",
        "warnings":[
          "String warning 1",
          "String warning 2"
        ],
        "crop_analysis": {{
          "confidence_score": 75,
          "recommended_crops":[
            "sugarcane",
            "rice",
            "String alternate crop"
          ],
          "non_recommended_crops":[
            "wheat",
            "String crop to avoid"
          ],
          "is_suitable_for_farming": true
        }},
        "actions": [
          "String action 1",
          "String action 2"
        ]
      }}
    }}
    """

    try:
        # Enforcing JSON format at the model level prevents text/markdown wrap
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)

    except Exception as e:
        print(f"❌ Gemini Generation Error: {e}")
        return None

# ------------------ LISTENER ------------------

def listen_to_nasa_for_roi():
    nasa_ref = db.collection('hackathon').document('PCCE2026') \
                 .collection('NASAreport').document(f"{em}_NASAreport")

    def on_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            if not doc.exists:
                continue

            print("\n💰 [TRIGGER] New NASA Data affecting Market Value & Crop Viability...")

            nasa_data = doc.to_dict()

            # Get Farm Profile Input
            farm_input_ref = db.collection('hackathon').document('PCCE2026') \
                         .collection('future_pred').document(f"{em}_input")
            farm_ref_doc = farm_input_ref.get()
            farm_profile = farm_ref_doc.to_dict() if farm_ref_doc.exists else {}

            # Run combined AI Analysis
            result = run_financial_and_crop_optimizer(nasa_data, farm_profile)

            if result:
                try:
                    # 1. Write to Financial Forecasting (Matching exact schema)
                    financial_payload = {
                        "nasa_trigger_id": doc.id,
                        "timestamp": datetime.now().isoformat(),
                        "analysis": result["financial_analysis"]
                    }
                    db.collection('hackathon').document('PCCE2026') \
                        .collection('financial_forecasting').document(f"{em}_latest") \
                        .set(financial_payload)

                    # 2. Write to Future Pred Output (Matching exact schema)
                   
                   

                    # 3. Mark input document as processed (Optional but seen in your DB tree)
                    if farm_ref_doc.exists:
                        farm_input_ref.update({"processed": True})

                    print(f"✅ Operations Success:")
                    print(f"   💰 ROI Updated: {result['financial_analysis']['roi_analysis']['roi_percentage']}%")
                    print(f"   🌾 Crop Confidence Score: {result['future_pred_output']['crop_analysis']['confidence_score']}/100")

                except Exception as e:
                    print(f"❌ Firestore Write Error: {e}")

    nasa_ref.on_snapshot(on_snapshot)

# ------------------ MAIN ------------------

if __name__ == "__main__":
    print("💎 Master ROI & Crop Forecasting Engine Online...")
    listen_to_nasa_for_roi()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")