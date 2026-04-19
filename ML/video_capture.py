import cv2
import time
import requests
import uuid
import os
import sys
import cloudinary
import cloudinary.uploader
import firebase_admin
from firebase_admin import credentials, firestore
# --- CONFIG ---
PREDICT_API = "http://127.0.0.1:8080/predict"

# 🔐 Cloudinary credentials (get from dashboard)
cloudinary.config(
    cloud_name="dqmwnci1k",
    api_key="531163372165284",
    api_secret="_WSbNtv9J_O0CMtytnIHxeIQL4s"
)
from firebase_admin import credentials, firestore

# 🔐 Cloudinary credentials (get from dashboard)
cloudinary.config(
    cloud_name="dqmwnci1k",
    api_key="531163372165284",
    api_secret="_WSbNtv9J_O0CMtytnIHxeIQL4s"
)



cred = credentials.Certificate("../firebase/serviceAccountKey.json")  # your firebase json
firebase_admin.initialize_app(cred)
db = firestore.client()
em=os.getenv("email")
def save_to_firebase(image_url):
    try:
        print("💾 Saving URL to Firebase...")

        db.collection("hackathon")\
          .document("PCCE2026")\
          .collection("videos")\
          .document(f"{em}_images")\
          .collection("images")\
          .document(str(uuid.uuid4()))\
          .set({
              "url": image_url,
              "timestamp": time.time()
          })

        print("✅ Saved to Firebase")

    except Exception as e:
        print("❌ Firebase error:", e)

# --- Upload to Cloudinary ---
def upload_to_cloudinary(file_path):
    try:
        print(f"📤 Uploading {file_path} to Cloudinary...")

        response = cloudinary.uploader.upload(file_path)

        image_url = response.get("secure_url")

        if not image_url:
            print("❌ Upload failed:", response)
            return None

        return image_url

    except Exception as e:
        print("❌ Upload error:", e)
        return None


# --- Call Prediction API ---
def call_prediction_api(image_url):
    try:
        print(f"🤖 Sending to prediction API: {image_url}")

        res = requests.post(
            PREDICT_API,
            json={"image_url": image_url},
            timeout=30
        )

        res.raise_for_status()
        print("✅ Prediction result:", res.json())

    except Exception as e:
        print("❌ Prediction API error:", e)


# --- Capture Image ---
def capture_once(cap):
    ret, frame = cap.read()
    if not ret:
        raise Exception("Camera capture failed")

    filename = f"image_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(filename, frame)

    print(f"📸 Captured: {filename}")
    return filename


# --- MAIN LOOP ---
def run_continuous():
    print("📷 Initializing camera...")

    if sys.platform == "darwin":
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open camera")
        sys.exit(1)

    INTERVAL = 30 * 60  # 30 minutes
    next_run = time.time()

    print("🚀 Service started (Cloudinary mode)")
    print("⏹ Press Ctrl+C to stop")

    while True:
        try:
            now = time.time()

            if now >= next_run:
                print("\n--- 📸 New Capture Cycle ---")

                file_path = None

                try:
                    # 1. Capture
                    file_path = capture_once(cap)

                    # 2. Upload
                    image_url = upload_to_cloudinary(file_path)

                    # 3. Predict
                    if image_url:
                        print("✅ Uploaded URL:", image_url)

                  
                        save_to_firebase(image_url)

                
                        call_prediction_api(image_url)
                    else:
                        print("❌ Skipping prediction (upload failed)")

                except Exception as e:
                    print("⚠️ Cycle error:", e)

                finally:
                    # Cleanup
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                        print("🗑️ Deleted local file")

                next_run = now + INTERVAL
                print(f"⏭ Next run at: {time.ctime(next_run)}")

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n🛑 Stopping service...")
            break

        except Exception as e:
            print("⚠️ Critical error:", e)
            print("🔁 Retrying in 10 seconds...")
            time.sleep(10)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera released. Service stopped.")


# --- ENTRY POINT ---
if __name__ == "__main__":
    run_continuous()