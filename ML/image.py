import cv2
import time
import uuid
import os
import cloudinary
import cloudinary.uploader
import firebase_admin
from firebase_admin import credentials, firestore

# --- CONFIG ---
PREDICT_API = "http://127.0.0.1:8080/predict"


cloudinary.config(
    cloud_name="dqmwnci1k",
    api_key="531163372165284",
    api_secret="_WSbNtv9J_O0CMtytnIHxeIQL4s"
)


cred = credentials.Certificate("../firebase/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


RECORD_DURATION = 60   
FPS = 20.0



cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Camera not accessible")
    exit()

frame_width = int(cam.get(3))
frame_height = int(cam.get(4))

while True:
    video_id = str(uuid.uuid4())

    avi_file = f"{video_id}.avi"
    mp4_file = f"{video_id}.mp4"

    # ✅ RECORD IN AVI (stable)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(avi_file, fourcc, FPS, (frame_width, frame_height))

    print("Recording started...")
    start_time = time.time()

    while int(time.time() - start_time) < RECORD_DURATION:
        ret, frame = cam.read()
        if not ret:
            break

        out.write(frame)
        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    out.release()
    print("Recording finished")

    # ✅ CONVERT TO MP4 USING FFMPEG
    print("Converting to MP4...")
    convert_cmd = f"ffmpeg -y -i {avi_file} -vcodec libx264 -acodec aac {mp4_file}"
    os.system(convert_cmd)

    # -------- UPLOAD TO CLOUDINARY -------- #
    try:
        print("Uploading to Cloudinary...")
        response = cloudinary.uploader.upload(
            mp4_file,
            resource_type="video"
        )

        video_url = response["secure_url"]
        print("Uploaded:", video_url)
        os.remove(mp4_file)  # Cleanup MP4 after upload
        os.remove(avi_file)  # Cleanup AVI after upload
        em=os.getenv("email")
        # ✅ SAVE TO FIREBASE
        db.collection("hackathon")\
          .document("PCCE2026")\
          .collection("videos")\
          .document(f"{em}_videos")\
          .set({
              "url": video_url,
              "timestamp": time.time()
          })

        print("Saved to Firebase")

    except Exception as e:
        print("Upload error:", e)

    # ✅ CLEANUP FILES
    if os.path.exists(avi_file):
        os.remove(avi_file)

    if os.path.exists(mp4_file):
        os.remove(mp4_file)

    print("Waiting for next recording...\n")