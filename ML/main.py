import firebase_admin
from firebase_admin import credentials, firestore
import requests
import json
import time
import traceback
from NASA_cords import fetch_nasa, process_nasa,get_nasa_data, analyze_with_gemini, store_to_firebase



cred = credentials.Certificate("/Users/vedeekaparab/Desktop/GDG-DroneML/firebase/serviceAccountKey.json")

db = firestore.client()


def process_user_doc(doc):
    data = doc.to_dict()

    if not data:
        return


    email = data.get("email", "").strip().lower()

    if email != "ved@gmail.com":
        print(f"❌ Skipping user: '{email}'")
        return

    if email != "ved@gmail.com":
        print(f"❌ Skipping user: {email}")
        return

    print(f"\n📍 Processing user: {data.get('email')}")
    # ✅ get coordinates from DB
    location = data.get("farmDetails", {}).get("location")

    if not location:
        print("❌ No location field")
        return

    try:
        lat_str, lon_str = location.split(",")

        lat = float(lat_str.strip())
        lon = float(lon_str.strip())

        print(f"📍 Coordinates: {lat}, {lon}")

    except Exception as e:
        print("❌ Invalid location format:", location)
        return
    

    if lat is None or lon is None:
        print("❌ Missing coordinates")
        return

    print(f"\n📍 Processing user: {data.get('email')} ({lat}, {lon})")

    try:
        nasa_data = get_nasa_data(lat, lon)
        print("✅ NASA:", nasa_data)

        ai_data = analyze_with_gemini(nasa_data, "wheat")
        print("✅ GEMINI:", json.dumps(ai_data, indent=2))

        store_to_firebase(nasa_data, ai_data, email)

    except Exception as e:
        print("❌ PIPELINE FAILED")
        traceback.print_exc()


# 🔁 FIRESTORE LISTENER
def on_snapshot(col_snapshot, changes, read_time):
    print("\n🔥 Change detected in Firestore")

    for change in changes:
        if change.type.name in ["ADDED", "MODIFIED"]:
            doc = change.document
            print(f"📄 Changed doc: {doc.id}")
            process_user_doc(doc)


if __name__ == "__main__":
    print("🚀 Listening to Firestore changes...")

    # 🔁 CHANGE THIS PATH to your actual collection
    col_ref = db.collection("hackathon").document("PCCE2026").collection("users")

    # 👂 Start listening
    col_ref.on_snapshot(on_snapshot)

    # ⛔ Keep script alive
    while True:
        time.sleep(1)