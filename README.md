# 🌿 GDG-DroneML — AI-Powered Drone Farm Intelligence System

> An end-to-end agricultural intelligence platform . Combines drone-captured
> imagery, NASA satellite weather data, computer vision, and Gemini AI to detect
> plant diseases, predict crop yields, and generate real-time farm alerts — all
> synced to Firebase Firestore.

---

## 📋 Table of Contents

- [System Overview](#-system-overview)
- [Architecture](#-architecture)
- [ML Files & Modules](#-ml-files--modules)
- [Model Choice: Why ResNet-18?](#-model-choice-why-resnet-18)
- [Why Gemini AI?](#-why-gemini-ai)
- [Dataset](#-dataset)
- [Firebase Firestore Structure](#-firebase-firestore-structure)
- [Setup & Installation](#-setup--installation)
- [Environment Variables](#-environment-variables)
- [Running the System](#-running-the-system)

---

## 🌐 System Overview

The platform addresses three core agricultural challenges through a fully
automated pipeline:

| Challenge                      | Solution                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| 🦠 Plant Disease Detection     | Ensemble of ResNet-18 models trained on PlantVillage dataset |
| 🌦️ Weather-Aware Crop Planning | NASA POWER API data analyzed by Gemini AI                    |
| 💰 Financial Forecasting       | ROI modeling triggered by real-time NASA satellite updates   |

The drone captures images every 30 minutes → images are uploaded to Cloudinary →
the prediction API runs the ensemble → results + Gemini treatment advice are
saved to Firestore → the Flutter mobile app displays everything in real time.

---

## 🏗️ Architecture

```
Drone Camera
    │
    ▼
video_capture.py / image.py
    │  (captures frame every 30 min)
    │
    ▼
Cloudinary (image CDN)
    │  (public image URL)
    │
    ▼
detect_disease.py  ◄──── Flask API (/predict)
    │
    ├── ResNet-18 Ensemble (4 best-epoch models)
    │       └── Majority vote → predicted disease
    │
    └── Gemini Flash ──► treatment advice JSON
    │
    ▼
Firebase Firestore
    ├── prediction/{userId}_disease
    ├── NASAreport/{userId}_NASAreport   ◄── NASA_cords.py / main.py
    ├── future_pred/{userId}_output       ◄── future_pred.py
    ├── crop_alerts/{userId}_output       ◄── crop_alert.py
    └── financial_forecasting/{userId}_latest ◄── crop_yield.py
```

---

## 📂 ML Files & Modules

### 1. `model_disease.py` — Model Training

**Purpose:** Trains the plant disease classification model on the PlantVillage
dataset using transfer learning.

**What it does:**

- Loads the **PlantVillage** dataset (supports both pre-split `train/val`
  folders or auto-split 80/20).
- Applies data augmentation: `RandomResizedCrop`, `RandomHorizontalFlip`, and
  ImageNet normalization.
- Fine-tunes a **ResNet-18** backbone using `SGD` optimizer with momentum.
- Saves the **best model** (`plant_disease_detector_best_model_epoch_N.pth`) and
  **epoch checkpoints** (`plant_disease_detector_checkpoint_epoch_N.pth`) after
  every validation epoch.
- Plots training/validation loss and accuracy curves.

**Key Config:**

| Parameter     | Value              |
| ------------- | ------------------ |
| Image Size    | 128×128            |
| Batch Size    | 32                 |
| Epochs        | 10                 |
| Learning Rate | 0.001              |
| Optimizer     | SGD (momentum=0.9) |
| Loss Function | CrossEntropyLoss   |

---

### 2. `detect_disease.py` — Inference API

**Purpose:** Flask REST API that runs the disease detection ensemble and returns
a prediction + Gemini treatment advice.

**Endpoint:** `POST /predict`

```json
{
    "image_url": "https://example.com/leaf.jpg",
    "user_id": "user@email.com"
}
```

**What it does:**

- Loads **all 4 best-epoch `.pth` model checkpoints** at startup (ensemble of
  epochs 0, 1, 2, 5).
- Downloads the image from the provided URL, runs it through every loaded model.
- Uses **majority voting** across model predictions (`EnsembleVoter`) to
  determine the final disease class.
- Calls **Gemini Flash** to generate structured JSON treatment advice (symptoms,
  causes, prevention, treatment categories).
- Saves the full prediction record to Firestore and returns the complete result
  as JSON.

**Classes Detected (15 total):**

| Plant            | Conditions                                                                                                                                         |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🫑 Pepper (Bell) | Bacterial Spot, Healthy                                                                                                                            |
| 🥔 Potato        | Early Blight, Late Blight, Healthy                                                                                                                 |
| 🍅 Tomato        | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus, Healthy |

---

### 3. `NASA_cords.py` — NASA Weather Fetcher

**Purpose:** Fetches real-time environmental data for a given GPS coordinate
using the **NASA POWER API** and runs initial Gemini crop analysis.

**What it does:**

- Queries NASA POWER API for temperature, humidity, precipitation, and solar
  radiation.
- Tries hourly data first, falls back to daily (retries last 5–12 days to get
  valid data).
- Sends environmental readings to **Gemini Flash** for crop health analysis.
- Stores the combined NASA + AI report in Firestore under
  `NASAreport/{email}_NASAreport`.

**NASA Parameters Fetched:**

| Parameter           | Description                     |
| ------------------- | ------------------------------- |
| `T2M`               | 2-metre air temperature (°C)    |
| `RH2M`              | 2-metre relative humidity (%)   |
| `PRECTOTCORR`       | Precipitation corrected (mm/hr) |
| `ALLSKY_SFC_SW_DWN` | Solar irradiance (W/m²)         |

---

### 4. `main.py` — Firestore NASA Trigger

**Purpose:** Listens to Firestore for new/updated user documents and
automatically runs the NASA → Gemini pipeline for that user's farm coordinates.

**What it does:**

- Watches the `hackathon/PCCE2026/users` Firestore collection for `ADDED` or
  `MODIFIED` events.
- Extracts latitude and longitude from the user's `farmDetails` sub-document.
- Calls `get_nasa_data()` + `analyze_with_gemini()` + `store_to_firebase()` from
  `NASA_cords.py`.

---

### 5. `future_pred.py` — Crop Suitability Advisor

**Purpose:** Listens for user-submitted crop plans and runs Gemini AI to assess
suitability given current NASA weather conditions.

**What it does:**

- Watches `hackathon/PCCE2026/future_pred/{email}_input` for new crop input
  data.
- Fetches the latest NASA report from Firestore.
- Sends both datasets to **Gemini Flash** which returns:
  - Crop suitability score and confidence (0–100)
  - Recommended vs. non-recommended crops
  - Weather impact analysis (temperature effect, rainfall effect, risk level)
  - Actionable farming recommendations and warnings
- Saves the AI response to `future_pred/{email}_output` and marks the input as
  `processed: true` to prevent duplicate runs.

**Trigger:** Firestore document change (user submits crop plan in the app)

---

### 6. `crop_yield.py` — Financial & Yield Optimizer

**Purpose:** NASA-triggered engine that calculates both **financial ROI** and
**physical crop viability** whenever new NASA data arrives.

**What it does:**

- Watches `NASAreport/{email}_NASAreport` for new NASA data snapshots.
- Sends NASA data + farm profile to **Gemini Flash** which returns:
  - **Financial Analysis:** ROI %, input costs, market price/kg, expected yield
    (kg/acre), projected profit, market forecast, cost-saving opportunities,
    efficiency score.
  - **Future Prediction:** crop viability summary, recommended/non-recommended
    crops, confidence score, farm suitability flag, warnings, and immediate
    actions.
- Writes results to `financial_forecasting/{email}_latest`.

**Trigger:** Any change to the NASA report document in Firestore

---

### 7. `crop_alert.py` — Real-Time NASA Alert Engine

**Purpose:** Runs a **Digital Twin simulation** every time new NASA data arrives
— comparing new conditions against the farm's last strategy.

**What it does:**

- Watches `NASAreport/{email}_NASAreport` for NASA data updates.
- Uses a **Reinforcement Learning framing** — Gemini explicitly validates or
  updates the previous strategy based on new data.
- Generates three simulation scenarios (Conservative / Balanced / Aggressive)
  with risk, yield, and profit estimates.
- Saves alert data to `crop_alerts/{email}_output` and updated financials to
  `financial_forecasting/{email}_latest`.

**Alert Levels:** `Low` / `Medium` / `High` (based on environmental severity)

---

### 8. `video_capture.py` — Drone Image Capture Service

**Purpose:** Continuous service that captures a photo from the drone/camera
every 30 minutes, uploads to Cloudinary, and triggers the disease prediction
API.

**What it does:**

- Opens the device camera (uses `CAP_AVFOUNDATION` on macOS for compatibility).
- Every 30 minutes: captures a frame → uploads to Cloudinary → calls `/predict`
  API → saves image URL to Firebase.
- Gracefully handles camera errors with 10-second retry logic.

**Interval:** 30 minutes (`INTERVAL = 30 * 60`)

---

### 9. `image.py` — Continuous Video Recorder

**Purpose:** Records 60-second video clips continuously, converts to MP4,
uploads to Cloudinary, and saves URLs to Firebase.

**What it does:**

- Records video at 20 FPS into `.avi` format using the `XVID` codec.
- Converts each clip to `.mp4` using `ffmpeg` (`libx264` codec).
- Uploads the MP4 to Cloudinary and stores the URL in Firestore.
- Cleans up both `.avi` and `.mp4` files after successful upload.

**Recording Duration:** 60 seconds per clip

---

### 10. `system_change_file.py` — ENV Configuration GUI

**Purpose:** A simple PyQt5 desktop GUI to update the `.env` file email without
needing to edit it manually.

**What it does:**

- Reads the current `.env` file at startup and pre-fills the email field.
- On "Save", writes the updated email back to `.env`.
- Used to switch between farmer accounts without restarting all services.

---

## 🧠 Model Choice: Why ResNet-18?

The project uses **ResNet-18 with Transfer Learning** for plant disease
classification. Here's the reasoning:

### ✅ Advantages for This Use Case

| Factor                 | Why ResNet-18 Wins                                                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Speed**              | Smallest ResNet variant — runs fast on CPU. Crucial for real-time drone deployments without a GPU.                                 |
| **Accuracy**           | Pre-trained on ImageNet gives strong feature extraction for leaf textures, colors, and disease patterns.                           |
| **Transfer Learning**  | Fine-tuning only the final `fc` layer (feature extraction mode) prevents overfitting on the relatively small PlantVillage dataset. |
| **Size**               | ~44MB per checkpoint — manageable for deployment.                                                                                  |
| **Ensemble Stability** | Running 4 epoch checkpoints as an ensemble via **majority voting** significantly reduces false positives.                          |

### Why Not MobileNetV2 or EfficientNet-B0?

The training script includes commented-out options for both. ResNet-18 was
chosen because:

- Its residual connections handle 128×128 images without information loss.
- It was most stable during training on available hardware.
- The accuracy difference on PlantVillage is marginal, but ResNet-18's simpler
  architecture makes the ensemble more predictable and debuggable.

### Ensemble Strategy

Rather than using a single model, **4 saved best-epoch checkpoints** (epochs 0,
1, 2, 5) are all loaded simultaneously. Every inference runs through all 4
models and **majority voting** determines the final class. This:

- Reduces variance from any individual training run.
- Captures disease patterns learned at different stages of training.
- Provides a built-in confidence signal — unanimous agreement = high confidence.

---

## 💡 Why Gemini AI?

**Gemini Flash** (`gemini-3-flash-preview`) is used across 4 modules for
different tasks:

| Module                            | Gemini Role                                                                               |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| `detect_disease.py`               | Generate treatment advice (symptoms, causes, prevention, fungicides) for detected disease |
| `NASA_cords.py`                   | Analyze environmental conditions → crop health assessment + carbon metrics                |
| `future_pred.py`                  | Crop suitability advisor — recommend/warn about specific crops given weather              |
| `crop_alert.py` + `crop_yield.py` | Digital Twin simulation — financial ROI, yield forecasting, alert generation              |

### Why Gemini Flash Specifically?

- **Speed:** Flash variant is optimized for low-latency, critical for real-time
  Firestore listeners.
- **JSON Mode:** Supports `response_mime_type: "application/json"` for
  structured, parseable output without markdown wrapping.
- **Gemini 3 Flash:** Latest generation model with improved reasoning for
  complex agricultural domain prompts.
- **Google Ecosystem:** Seamless integration with Firebase/GCP.

---

## 📊 Dataset

### PlantVillage Dataset

The model is trained on the **PlantVillage** dataset stored in the
`PlantVillage/` directory.

| Crop          | Disease Classes                                                                                                                                    |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pepper (Bell) | Bacterial Spot, Healthy                                                                                                                            |
| Potato        | Early Blight, Late Blight, Healthy                                                                                                                 |
| Tomato        | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus, Healthy |

**Total Classes:** 15 (including 3 healthy variants)

Dataset is organized as class-labeled subdirectories — compatible with PyTorch's
`ImageFolder` loader. If no `train/val` split exists, the script auto-creates an
**80/20 split**.

---

## 🔥 Firebase Firestore Structure

```
hackathon/
└── PCCE2026/
    ├── users/                           ← User profiles with farmDetails (lat/lon)
    ├── NASAreport/
    │   └── {email}_NASAreport           ← NASA weather + Gemini crop analysis
    ├── prediction/
    │   └── {userId}_disease             ← Disease prediction results + Gemini advice
    ├── future_pred/
    │   ├── {email}_input                ← User-submitted crop plan
    │   └── {email}_output               ← AI crop suitability analysis
    ├── crop_alerts/
    │   └── {email}_output               ← Digital Twin alert (level, simulations, actions)
    ├── financial_forecasting/
    │   └── {email}_latest               ← ROI, yield, market forecast, efficiency score
    └── videos/
        ├── {email}_videos               ← Video URLs
        └── {email}_images/images/       ← Per-image capture URLs
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- `ffmpeg` installed (for `image.py` video conversion)
- Firebase project with Firestore enabled + service account key
- Cloudinary account
- Google Gemini API key
- NASA POWER API (free, no key required)

### Install Dependencies

```bash
pip install torch torchvision
pip install flask
pip install firebase-admin
pip install google-generativeai
pip install python-dotenv
pip install requests pillow opencv-python
pip install cloudinary
pip install PyQt5
pip install matplotlib numpy
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
email="your_farmer_email@example.com"
GEMINI_API_KEY="your_gemini_api_key_here"
```

Place your Firebase service account JSON at:

```
firebase/serviceAccountKey.json
```

> ⚠️ **Never commit `.env` or `serviceAccountKey.json`** — both are in
> `.gitignore`.

---

## 🚀 Running the System

Each script is an independent service. Start them in separate terminals:

```bash
# 1. Disease Detection API
cd ML && python detect_disease.py
# → Available at http://0.0.0.0:8080/predict

# 2. Drone Image Capture (picks up every 30 min)
cd ML && python video_capture.py

# 3. NASA Pipeline Trigger (listens to Firestore user changes)
cd ML && python main.py

# 4. Crop Suitability Advisor (listens for crop plan input)
cd ML && python future_pred.py

# 5. Financial & Yield Optimizer (listens for NASA updates)
cd ML && python crop_yield.py

# 6. Real-Time Alert Engine (Digital Twin on NASA update)
cd ML && python crop_alert.py

# 7. Update Farmer Email (PyQt5 GUI)
cd ML && python system_change_file.py
```

---

## 📌 Model Checkpoints

| File                                            | Description                                                              |
| ----------------------------------------------- | ------------------------------------------------------------------------ |
| `plant_disease_detector_best_model_epoch_N.pth` | Best validation accuracy snapshot at epoch N (state dict only, ~44MB)    |
| `plant_disease_detector_checkpoint_epoch_N.pth` | Full training checkpoint at epoch N (model + optimizer + history, ~44MB) |

The inference API (`detect_disease.py`) automatically loads all `best_model`
files via glob pattern for ensemble inference.

---

## 🏆 Built For

**GDG PCCE 2026 Hackathon** — _AI for Agriculture Track_

Combining **Edge AI** (drone camera + local CNN inference) · **Satellite Data**
(NASA POWER API) · **Generative AI** (Google Gemini Flash) · **Cloud
Infrastructure** (Firebase Firestore + Cloudinary)
