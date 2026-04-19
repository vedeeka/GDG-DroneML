

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import glob
import logging
import json # Import json for parsing Gemini's output
from collections import Counter
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from io import BytesIO

# Flask imports for the API
from flask import Flask, request, jsonify, make_response

# Firebase imports (Firestore specific)
import firebase_admin
from firebase_admin import credentials, firestore

# Gemini API imports
import google.generativeai as genai

# ================== CONFIGURATION ==================
@dataclass
class Config:
    """Configuration constants for the predictor."""
    INFERENCE_IMAGE_SIZE: int = 128  # Must match training size
    IMAGENET_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Disease class names - UPDATE THIS WITH YOUR ACTUAL CLASSES IN THE CORRECT ORDER
    CLASS_NAMES: List[str] = None

    # Firebase Configuration (Firestore specific)
    FIREBASE_CREDENTIALS_PATH: str = "/Users/vedeekaparab/Desktop/GDG-DroneML/firebase/serviceAccountKey.json" # <--- UPDATED PATH
    # FIREBASE_DB_URL is not needed for Firestore initialized with credentials
    FIREBASE_BASE_COLLECTION_PATH: str = "hackathon" # Base path for Firestore collections

    # Gemini API Configuration
    GEMINI_API_KEY = "AQ.Ab8RN6K5eKwbONu4z7NmuziiHIGXPGljDiycwOfpPImVHjI_wA" # Recommended: set as environment variable

    def __post_init__(self):
        if self.CLASS_NAMES is None:
            self.CLASS_NAMES = [
                "Pepper__bell___Bacterial_spot",
                "Pepper__bell___healthy", # This is a healthy class
                "Potato___Early_blight",
                "Potato___healthy",       # This is a healthy class
                "Potato___Late_blight",
                "Tomato___Target_Spot",
                "Tomato___Tomato_mosaic_virus",
                "Tomato___Tomato_YellowLeaf_Curl_Virus",
                "Tomato__Bacterial_spot",
                "Tomato__Early_blight",
                "Tomato__healthy",        # This is a healthy class
                "Tomato__Late_blight",
                "Tomato__Leaf_Mold",
                "Tomato__Septoria_leaf_spot",
                "Tomato__Spider_mites_Two_spotted_spider_mite"
            ]
        
        if not self.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not found in environment variables. Gemini features may not work.")
            # For development, you can uncomment the line below and paste your key directly.
            # self.GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE" # REMOVE IN PRODUCTION


# ================== LOGGING SETUP ==================
def setup_logger(name: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger

logger = setup_logger(__name__)

# ================== PREDICTION DATA STRUCTURES ==================
@dataclass
class ModelPrediction:
    """Container for a single model's prediction."""
    model_name: str
    predicted_class: str
    confidence: float
    class_index: int
    
    def __str__(self) -> str:
        return (f"Model: {self.model_name:<45} | "
                f"Disease: {self.predicted_class:<40} | "
                f"Confidence: {self.confidence:.4f}")


@dataclass
class EnsemblePrediction:
    """Container for ensemble voting results."""
    predicted_class: str
    vote_count: int
    total_votes: int
    percentage: float
    vote_breakdown: Dict[str, int]
    
    def __str__(self) -> str:
        return (f"Disease: {self.predicted_class}\n"
                f"Votes: {self.vote_count}/{self.total_votes} ({self.percentage:.2f}%)")


# ================== MODEL LOADING ==================
class ModelLoader:
    """Handles loading and validation of ResNet18 models."""
    
    def __init__(self, device: torch.device, config: Config):
        self.device = device
        self.config = config
        logger.debug(f"ModelLoader initialized for device: {device}")
    
    def load_model(self, model_path: str) -> Tuple[nn.Module, int]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.debug(f"Loading model from: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint '{os.path.basename(model_path)}': {e}")
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        num_classes = self._infer_num_classes(state_dict, model_path)
        model = self._build_model(num_classes)
        
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.debug(f"Model '{os.path.basename(model_path)}' loaded with strict=True")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed for '{os.path.basename(model_path)}', attempting non-strict: {e}")
            try:
                model.load_state_dict(state_dict, strict=False)
                logger.debug(f"Model '{os.path.basename(model_path)}' loaded with strict=False")
            except Exception as e2:
                raise RuntimeError(f"Failed to load state dict for '{os.path.basename(model_path)}': {e2}")
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully: {os.path.basename(model_path)} "
                   f"({num_classes} classes)")
        return model, num_classes
    
    def _infer_num_classes(self, state_dict: dict, model_path: str) -> int:
        classifier_layers = ["fc", "classifier", "final_layer"]
        
        for layer_name in classifier_layers:
            bias_key = f"{layer_name}.bias"
            weight_key = f"{layer_name}.weight"
            
            if bias_key in state_dict:
                num_classes = state_dict[bias_key].shape[0]
                logger.debug(f"Inferred {num_classes} classes from {bias_key} in {os.path.basename(model_path)}")
                return num_classes
            
            if weight_key in state_dict:
                num_classes = state_dict[weight_key].shape[0]
                logger.debug(f"Inferred {num_classes} classes from {weight_key} in {os.path.basename(model_path)}")
                return num_classes
        
        for key in sorted(state_dict.keys()):
            if "bias" in key and state_dict[key].ndim == 1:
                if any(k in key for k in classifier_layers):
                    num_classes = state_dict[key].shape[0]
                    logger.debug(f"Inferred {num_classes} classes from general bias key '{key}' in {os.path.basename(model_path)}")
                    return num_classes
            elif "weight" in key and state_dict[key].ndim == 2:
                 if any(k in key for k in classifier_layers):
                    num_classes = state_dict[key].shape[0]
                    logger.debug(f"Inferred {num_classes} classes from general weight key '{key}' in {os.path.basename(model_path)}")
                    return num_classes

        raise ValueError(
            f"Could not infer number of classes from model state_dict for '{os.path.basename(model_path)}'. "
            f"Ensure the final classification layer's 'bias' or 'weight' parameter is present and identifiable."
        )
    
    def _build_model(self, num_classes: int) -> nn.Module:
        model = models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        logger.debug(f"Built ResNet18 with {num_classes} output classes")
        return model


# ================== INFERENCE ==================
class ImagePredictor:
    """Handles image loading and single-image inference."""
    
    def __init__(self, device: torch.device, config: Config):
        self.device = device
        self.config = config
        self.transform = self._build_transform()
    
    def _build_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((self.config.INFERENCE_IMAGE_SIZE, 
                              self.config.INFERENCE_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.config.IMAGENET_MEAN, 
                               self.config.IMAGENET_STD)
        ])
    
    def predict(self, image: Image.Image, model: nn.Module, 
                class_names: List[str]) -> Tuple[str, float, int]:
        
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)
        
        confidence_val = confidence.item()
        class_idx_val = class_idx.item()
        
        if 0 <= class_idx_val < len(class_names):
            class_name = class_names[class_idx_val]
        else:
            raise ValueError(
                f"Predicted class index {class_idx_val} out of bounds for model {os.path.basename(model.state_dict)}. "
                f"Expected 0-{len(class_names)-1}. "
                f"Verify CLASS_NAMES in config matches this model's output classes."
            )
        
        logger.debug(f"Prediction: {class_name} (confidence: {confidence_val:.4f})")
        return class_name, confidence_val, class_idx_val


# ================== ENSEMBLE VOTING ==================
class EnsembleVoter:
    """Handles ensemble predictions via majority voting."""
    
    @staticmethod
    def vote(predictions: List[ModelPrediction]) -> EnsemblePrediction:
        if not predictions:
            raise ValueError("No predictions to ensemble")
        
        class_names = [p.predicted_class for p in predictions]
        vote_counts = Counter(class_names)
        
        most_common = vote_counts.most_common(1)
        if not most_common:
            raise ValueError("Failed to compute voting results for ensemble")
        
        top_class, vote_count = most_common[0]
        total_votes = len(predictions)
        percentage = (vote_count / total_votes) * 100
        
        logger.info(f"Ensemble voting: '{top_class}' received {vote_count}/{total_votes} votes")
        
        return EnsemblePrediction(
            predicted_class=top_class,
            vote_count=vote_count,
            total_votes=total_votes,
            percentage=percentage,
            vote_breakdown=dict(vote_counts)
        )


# ================== GEMINI INTEGRATION ==================
class GeminiAssistant:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API Key is not provided.")
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("Gemini model initialized.")

    def get_disease_solution(self, disease_name: str) -> Dict[str, str]:
        # --- THIS IS THE SECTION THAT HANDLES "HEALTHY" PLANTS ---
        if "healthy" in disease_name.lower():
            return {
                "diagnosis_summary": "The plant appears healthy.",
                "recommendation": "Continue with good horticultural practices: proper watering, fertilization, and pest monitoring.",
                "further_info": "A healthy plant indicates successful care. Regular inspections can prevent future issues.",
                "symptoms": "No symptoms detected.",
                "causes": "N/A",
                "prevention": "N/A",
                "treatment": "N/A"
            }
        # --- END OF "HEALTHY" PLANT HANDLING ---

        # --- THIS IS THE SECTION THAT QUERIES GEMINI FOR DISEASE SOLUTIONS ---
        prompt = (
            f"Provide a comprehensive solution and detailed information for the plant disease: '{disease_name}'. "
            "Structure the response in JSON format with the following keys:\n"
            "- 'diagnosis_summary': A brief summary of the disease.\n"
            "- 'symptoms': A list of common symptoms.\n"
            "- 'causes': Common causes or conditions favoring the disease.\n"
            "- 'prevention': Steps to prevent the disease.\n"
            "- 'treatment': Methods to treat the disease once it's present.\n"
            "- 'further_info': Any other relevant information, like environmental factors or specific host plants.\n"
            "Ensure the response is practical and actionable for a plant owner/farmer."
        )

        try:
            response = self.model.generate_content(prompt)
            text_response = response.text.strip()
            # Try to extract JSON from markdown if present
            if text_response.startswith("```json") and text_response.endswith("```"):
                text_response = text_response[7:-3].strip()
            
            solution_data = json.loads(text_response)
            logger.debug(f"Gemini provided solution for '{disease_name}'.")
            return solution_data
        except Exception as e:
            logger.error(f"Error getting Gemini solution for '{disease_name}': {e}")
            return {
                "diagnosis_summary": f"Could not retrieve detailed information for {disease_name}.",
                "recommendation": "Please consult a local agricultural expert.",
                "further_info": str(e),
                "symptoms": "N/A",
                "causes": "N/A",
                "prevention": "N/A",
                "treatment": "N/A"
            }
        # --- END OF DISEASE SOLUTION QUERY ---


# ================== MAIN PREDICTOR ==================
class PlantDiseasePredictor:
    """Main orchestrator for plant disease prediction and Firestore storage."""
    
    def __init__(self, config: Config, model_paths: List[str], verbose: bool = False):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.loader = ModelLoader(self.device, config)
        self.predictor = ImagePredictor(self.device, config)
        self.verbose = verbose
        
        self.models: List[Tuple[nn.Module, str]] = []
        self._load_all_models(model_paths)
        
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.info(f"PlantDiseasePredictor initialized on device: {self.device}")

        # Initialize Firebase (Firestore specific)
        self.db = None
        if not firebase_admin._apps: # Check if Firebase is already initialized
            try:
                cred = credentials.Certificate(self.config.FIREBASE_CREDENTIALS_PATH)
                firebase_admin.initialize_app(cred) # No databaseURL for Firestore init with credentials
                self.db = firestore.client()
                logger.info("Firebase Firestore initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Firebase Firestore: {e}. Predictions will not be saved to DB.")
        else:
            self.db = firestore.client()
            logger.info("Firebase Firestore already initialized.")
        
        # Initialize Gemini Assistant
        self.gemini_assistant = None
      
        try:
                self.gemini_assistant = GeminiAssistant(self.config.GEMINI_API_KEY)
        except ValueError as e:
                logger.error(f"Gemini API initialization failed: {e}. Gemini solutions will not be available.")
  


    def _load_all_models(self, model_paths: List[str]):
        logger.info(f"Pre-loading {len(model_paths)} models...")
        for i, model_path in enumerate(model_paths, 1):
            model_name = os.path.basename(model_path)
            try:
                model, num_classes = self.loader.load_model(model_path)
                
                if num_classes != len(self.config.CLASS_NAMES):
                    logger.error(
                        f"  CRITICAL: Model '{model_name}' has {num_classes} output classes, "
                        f"but `CLASS_NAMES` in config has {len(self.config.CLASS_NAMES)} entries. "
                        f"This model's predictions may be incorrect."
                    )
                self.models.append((model, model_name))
            except Exception as e:
                logger.error(f"  ✗ Failed to load model '{model_name}': {e}")
        logger.info(f"Finished pre-loading {len(self.models)} models.")
    
    def predict_image(self, image: Image.Image, 
                     image_identifier: str = "unknown_image") -> Tuple[List[ModelPrediction], Optional[EnsemblePrediction], Optional[ModelPrediction], Optional[Dict[str, str]]]:
        """
        Generate predictions for an image using multiple models and get Gemini solution.
        
        Returns:
            Tuple of (individual_predictions, ensemble_prediction, best_single_prediction, gemini_solution)
        """
        individual_predictions = []
        best_single = None
        max_confidence = -1
        gemini_solution = None # This will hold the Gemini response

        if not self.models:
            logger.error("No models were successfully loaded. Cannot perform prediction.")
            return [], None, None, None

        logger.info(f"Starting inference on {len(self.models)} pre-loaded models for image: {image_identifier}")
        
        for i, (model, model_name) in enumerate(self.models, 1):
            logger.debug(f"[{i}/{len(self.models)}] Processing model: {model_name}")
            
            try:
                class_name, confidence, class_idx = self.predictor.predict(
                    image, model, self.config.CLASS_NAMES
                )
                
                pred = ModelPrediction(
                    model_name=model_name,
                    predicted_class=class_name,
                    confidence=confidence,
                    class_index=class_idx
                )
                individual_predictions.append(pred)
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_single = pred
                
                logger.debug(f"  ✓ Prediction: {class_name} (confidence: {confidence:.4f})")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process model '{model_name}' for image '{image_identifier}': {e}")
        
        ensemble_pred = None
        if individual_predictions:
            try:
                ensemble_pred = EnsembleVoter.vote(individual_predictions)
                # --- THIS IS WHERE THE GEMINI ASSISTANT IS CALLED ---
                if ensemble_pred and self.gemini_assistant:
                    gemini_solution = self.gemini_assistant.get_disease_solution(ensemble_pred.predicted_class)
                # --- END OF GEMINI ASSISTANT CALL ---
            except Exception as e:
                logger.error(f"Failed to compute ensemble prediction or get Gemini solution for image '{image_identifier}': {e}")
        else:
            logger.warning("No successful individual predictions to form an ensemble.")
        
        return individual_predictions, ensemble_pred, best_single, gemini_solution
    
    def save_to_firestore(self, user_id: str, prediction_data: Dict):
        if not self.db:
            logger.error("Firestore not initialized. Cannot save data.")
            return None
        
        try:
            # Construct the Firestore path: hackathon/PCCE2026/prediction/disease/{user_id}/predictions
            doc_ref = self.db.collection("hackathon") \
                .document("PCCE2026") \
                .collection("prediction") \
                .document(user_id+"_disease")   # 👈 important

            doc_ref.set(prediction_data)
            logger.info(f"Prediction data saved to Firestore at {doc_ref.path}")
            return doc_ref.id # Return the unique document ID
        except Exception as e:
            logger.error(f"Failed to save prediction data to Firestore for user '{user_id}': {e}")
            return None


# ================== API SETUP ==================

app = Flask(__name__)

predictor_instance: Optional[PlantDiseasePredictor] = None
global_config: Optional[Config] = None
global_model_paths: List[str] = []
global_verbose: bool = False

def initialize_predictor():
    global predictor_instance, global_config, global_model_paths, global_verbose
    
    setup_logger(__name__, global_verbose)
    
    global_config = Config()
    
    # --- IMPORTANT: Ensure firebase_credentials.json path is correct ---
    if not os.path.exists(global_config.FIREBASE_CREDENTIALS_PATH):
        logger.error(f"Firebase credentials file not found at {global_config.FIREBASE_CREDENTIALS_PATH}. Please provide it.")
        sys.exit(1)

    default_models_pattern = "plant_disease_detector_best_model_epoch_*.pth" 
    potential_model_files = sorted(glob.glob(default_models_pattern))
    if not potential_model_files:
        potential_model_files = sorted(glob.glob("models/" + default_models_pattern))

    global_model_paths = potential_model_files
    
    if not global_model_paths:
        logger.warning(f"No models found matching pattern '{default_models_pattern}' or 'models/{default_models_pattern}'. "
                       "The API will not be able to perform predictions.")
            
    predictor_instance = PlantDiseasePredictor(global_config, global_model_paths, verbose=global_verbose)

initialize_predictor() # Call initialization when the script starts

@app.route('/')
def home():
    return jsonify({"status": "API is running", "message": "Send a POST request to /predict with an 'image_url' and optional 'user_id' in JSON body."})

@app.route('/predict', methods=['POST'])
def predict_from_url():
    if predictor_instance is None or predictor_instance.db is None:
        return make_response(jsonify({"error": "Predictor or Firestore not initialized. Check logs for issues."}), 500)

    data = request.get_json()
    if not data or 'image_url' not in data:
        return make_response(jsonify({"error": "Missing 'image_url' in JSON body."}), 400)
    
    image_url = data['image_url']
    em=os.environ.get("email")
    user_id = data.get('user_id', em) # Default to 'anonymous_user' if not provided
    
    logger.info(f"Received prediction request for user '{user_id}', image URL: {image_url}")

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        logger.debug(f"Image downloaded successfully from {image_url}")

        individual_preds, ensemble_pred, best_single, gemini_solution = predictor_instance.predict_image(image, image_url)
        
        prediction_record = {
            "image_url": image_url,
            "user_id": user_id, # Include user_id in the record
            "timestamp": datetime.now().isoformat(),
            "device": predictor_instance.device.type,
            "individual_predictions": [],
            "best_single_prediction": None,
            "ensemble_prediction": None,
            "gemini_solution": gemini_solution, # --- THIS INCLUDES THE GEMINI RESPONSE IN THE API OUTPUT ---
            "status": "success"
        }

        if individual_preds:
            prediction_record["individual_predictions"] = [asdict(p) for p in individual_preds]
        
        if best_single:
            prediction_record["best_single_prediction"] = asdict(best_single)
        
        if ensemble_pred:
            prediction_record["ensemble_prediction"] = asdict(ensemble_pred)
            prediction_record["ensemble_prediction"]["vote_breakdown"] = ensemble_pred.vote_breakdown
        
        # Save to Firestore
        firestore_doc_id = predictor_instance.save_to_firestore(user_id, prediction_record)
        if firestore_doc_id:
            prediction_record["firestore_document_id"] = firestore_doc_id
            logger.info(f"Prediction for user '{user_id}', image {image_url} saved to Firestore with ID: {firestore_doc_id}")
        else:
            prediction_record["status"] = "success_firestore_save_failed"
            logger.error(f"Failed to save prediction for user '{user_id}', image {image_url} to Firestore.")

        logger.info(f"Prediction successful for {image_url}. Ensemble result: {ensemble_pred.predicted_class if ensemble_pred else 'N/A'}")
        return jsonify(prediction_record)

    except requests.exceptions.Timeout:
        logger.error(f"Image download timed out for URL: {image_url}")
        return make_response(jsonify({"error": f"Image download timed out for {image_url}.", "status": "image_download_timeout"}), 408)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {image_url}: {e}")
        return make_response(jsonify({"error": f"Failed to download image: {e}", "status": "image_download_failed"}), 400)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during prediction for {image_url}.")
        return make_response(jsonify({"error": f"An internal server error occurred: {e}", "status": "internal_error"}), 500)

# ================== RUN THE API ==================
if __name__ == '__main__':
    logger.info("Starting Flask API. Access at http://127.0.0.1:8080/")
    app.run(debug=True, host='0.0.0.0', port=8080)