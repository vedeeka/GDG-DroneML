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
    raise ValueError("❌ Email not found in .env")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

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

# ------------------ ROI ENGINE ------------------

def run_financial_optimizer(nasa_data, farm_profile):
    """
    Core feature: ROI + financial decision engine
    """

    # 🔥 CLEAN DATA BEFORE JSON
    nasa_data_clean = make_json_serializable(nasa_data)
    farm_profile_clean = make_json_serializable(farm_profile)

    prompt = f"""
    You are a Financial Agricultural Consultant. 
    Analyze the NEW NASA Weather Data against the Farm Profile to produce a Cost-vs-Yield ROI Analysis.
    
    NASA DATA (Current Environment): {json.dumps(nasa_data_clean)}
    FARM PROFILE (Current Crop/Soil): {json.dumps(farm_profile_clean)}

    TASK:
    1. Calculate 'Yield Loss Risk' based on the new weather (0-100%).
    2. Estimate 'Cost of Maintenance' (Water/Fertilizer) required to survive this weather.
    3. Calculate ROI: (Expected Market Value - Input Costs) / Input Costs.
    4. Provide a 'Pivot vs. Persevere' recommendation.

    RETURN ONLY JSON:
    {{
      "financial_summary": "Short executive summary of profit outlook",
      "roi_analysis": {{
        "expected_yield_kg_per_acre": 0,
        "estimated_market_price_per_kg": 0,
        "input_costs_usd": 0,
        "projected_profit_usd": 0,
        "roi_percentage": 0
      }},
      "efficiency_score": 0, 
      "cost_saving_opportunities": [
        "example suggestion"
      ],
      "market_forecast": "Bullish/Bearish",
      "action_plan": "Specific financial move to make today"
    }}
    """

    try:
        response = model.generate_content(prompt)

        # 🔥 CLEAN GEMINI RESPONSE
        clean_text = re.sub(r"```json\n?|```", "", response.text).strip()

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            print("⚠️ Gemini returned invalid JSON:")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Financial Engine Error: {e}")
        return None

# ------------------ LISTENER ------------------

def listen_to_nasa_for_roi():
    nasa_ref = db.collection('hackathon').document('PCCE2026') \
                 .collection('NASAreport').document(f"{em}_NASAreport")

    def on_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            if not doc.exists:
                continue

            print("\n💰 [FINANCIAL TRIGGER] New NASA Data affecting Market Value...")

            nasa_data = doc.to_dict()

            # Get Farm Profile
            farm_ref = db.collection('hackathon').document('PCCE2026') \
                         .collection('future_pred').document(f"{em}_input").get()

            farm_profile = farm_ref.to_dict() if farm_ref.exists else {}

            # Run ROI Analysis
            analysis = run_financial_optimizer(nasa_data, farm_profile)

            if analysis:
                try:
                    db.collection('hackathon').document('PCCE2026') \
                        .collection('financial_forecasting').document(f"{em}_latest") \
                        .set({
                            "timestamp": datetime.now().isoformat(),
                            "analysis": analysis,
                            "nasa_trigger_id": doc.id
                        })

                    print(f"✅ ROI Updated: {analysis['roi_analysis']['roi_percentage']}%")
                    print(f"💡 Action: {analysis['action_plan']}")

                except Exception as e:
                    print(f"❌ Firestore Write Error: {e}")

    nasa_ref.on_snapshot(on_snapshot)

# ------------------ MAIN ------------------

if __name__ == "__main__":
    print("💎 ROI & Profit Forecasting Engine Online...")
    listen_to_nasa_for_roi()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")