"""
Prediction utilities for the Fresh vs Rotten Fruit Detection app.

This module contains ALL machine-learning logic (model loading,
preprocessing, prediction). It has no Streamlit code, so it can be tested
on its own and reused elsewhere.

Training pipeline this module mirrors (from CNN_Project.ipynb):
    1. Images loaded with image_dataset_from_directory(image_size=(224, 224))
       -> bilinear resize, float32, pixel range 0-255
    2. image / 255.0
    3. Binary label:  0 = Fresh, 1 = Rotten  (sigmoid output, threshold 0.5)
"""

from __future__ import annotations

import os
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
IMG_SIZE = (224, 224)            # (height, width) used during training
THRESHOLD = 0.5                  # sigmoid threshold: >= 0.5 -> Rotten
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
MODEL_FILENAME = "fruits_classification_model.keras"
# Hugging Face repository
HF_REPO_ID = "vishal953436/fruit-freshness-model"

# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class PredictionResult:
    """Everything the UI needs to display one prediction."""
    label: str              # "Fresh" or "Rotten"
    confidence: float       # % confidence in the predicted label
    fresh_prob: float       # % probability of Fresh
    rotten_prob: float      # % probability of Rotten
    raw_score: float        # raw sigmoid output (probability of Rotten, 0-1)

# --------------------------------------------------------------------------
# Model loading from Hugging Face
# --------------------------------------------------------------------------

def load_trained_model() -> tf.keras.Model:
    """
    Download the trained CNN from Hugging Face and load it.

    Hugging Face automatically caches the downloaded model,
    so it does not need to be downloaded for every prediction.
    """

    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
        )

        # Load the trained model for inference
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        return model

    except Exception as exc:
        raise RuntimeError(
            f"Unable to download or load the model from Hugging Face: {exc}"
        ) from exc



# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def load_image(uploaded_file) -> Image.Image:
    """
    Open an uploaded file as an RGB PIL image.

    Raises:
        ValueError: if the file is not a valid, readable image.
    """
    try:
        image = Image.open(uploaded_file)
        image.load()                          # forces full decode -> catches corrupt files
        image = ImageOps.exif_transpose(image)  # fixes sideways phone photos
        return image.convert("RGB")           # drops alpha channel / grayscale
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError(
            "The file could not be read as an image. It may be corrupted "
            "or in an unsupported format. Please upload a JPG, JPEG or PNG."
        ) from exc


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into the exact tensor the model was trained on.

    Steps (identical to training):
        RGB -> resize to 224x224 (bilinear) -> float32 -> divide by 255.0
        -> add batch dimension

    Returns:
        np.ndarray of shape (1, 224, 224, 3), dtype float32, values in [0, 1].
    """
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    # tf.image.resize is what image_dataset_from_directory uses internally
    # (bilinear), so this matches the training pipeline pixel-for-pixel.
    resized = tf.image.resize(array, IMG_SIZE).numpy()
    normalized = resized.astype(np.float32) / 255.0
    batch = np.expand_dims(normalized, axis=0)
    assert batch.shape == (1, IMG_SIZE[0], IMG_SIZE[1], 3)
    return batch


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_freshness(model: tf.keras.Model, processed_image: np.ndarray) -> PredictionResult:
    """
    Run the CNN and turn the sigmoid output into a readable result.

    The model outputs ONE number between 0 and 1 = probability of "Rotten".
        score >= 0.5 -> Rotten
        score <  0.5 -> Fresh
    """
    score = float(model.predict(processed_image, verbose=0)[0][0])

    rotten_prob = score * 100
    fresh_prob = (1 - score) * 100

    if score >= THRESHOLD:
        label, confidence = "Rotten", rotten_prob
    else:
        label, confidence = "Fresh", fresh_prob

    return PredictionResult(
        label=label,
        confidence=confidence,
        fresh_prob=fresh_prob,
        rotten_prob=rotten_prob,
        raw_score=score,
    )
