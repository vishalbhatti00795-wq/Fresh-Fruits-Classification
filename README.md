# 🍎 Fresh vs Rotten Fruit Detection

A Streamlit web app that uses a **custom CNN (TensorFlow/Keras)** to classify a fruit photo as **Fresh** or **Rotten**.

## Features
- Dashboard-style UI with sidebar navigation (Home, Fruit Detection, Model Information, Prediction History)
- Upload JPG / JPEG / PNG, preview, **Analyze Fruit**, and **Reset**
- Fresh/Rotten result card with confidence and both probabilities
- Session prediction history and CSV downloads (single report or full history)
- Graceful handling of a missing model file and corrupted or unsupported images

## Model
| Item | Value |
|---|---|
| Input | 224 × 224 × 3 RGB, pixels ÷ 255.0 |
| Layers | 3 × (Conv2D + MaxPool) → Flatten → Dense(128) → Dense(1, sigmoid) |
| Task | Binary: 0 = Fresh, 1 = Rotten (threshold 0.5) |
| Source data | 6 classes (fresh/rotten × apples, bananas, oranges), merged to 2 |
| Training / Validation / Test accuracy | 97.82% / 98.35% / 98.15% |
| Test loss | 0.0490 |

> ⚠️ **Limitations:** 98.15% is the measured accuracy on the dataset's **test split**, not a guarantee for individual or real-world images. The model only saw apples, bananas and oranges and does **not** identify fruit type.

## Project structure
```text
Fruit_Freshness_Detection/
├── app.py                          # Streamlit UI
├── fruits_classification_model.keras
├── requirements.txt
├── README.md
├── .gitattributes                  # Git LFS tracking for the model
├── .streamlit/config.toml          # Theme
├── utils/
│   ├── __init__.py
│   └── prediction.py               # Model loading, preprocessing, prediction
└── assets/
```

## Run locally (VS Code)
```bash
# 1. Open the Fruit_Freshness_Detection folder in VS Code, then open a terminal
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

# 2. Install (Python 3.10 - 3.12)
pip install -r requirements.txt

# 3. Start the app
streamlit run app.py
```
The app opens at http://localhost:8501.

## Test with real-world images
1. Open **Fruit Detection**, upload a photo, press **Analyze Fruit**.
2. Try a mix: clearly fresh fruit, clearly rotten fruit, different lighting and backgrounds, and phone photos.
3. Try edge cases: a non-fruit image, a blurry photo, a fruit the model never saw. Expect weaker results there.
4. Try a renamed text file (`fake.jpg`) to confirm the error message.
5. Check **Prediction History** and download the CSV.

## Deploy on Streamlit Community Cloud
The model is ~134 MB, above GitHub's 100 MB per-file limit, so choose one option:

**Option A: Git LFS (simplest)**
```bash
git lfs install
git lfs track "*.keras"      # already in .gitattributes
git add .gitattributes . && git commit -m "Initial commit" && git push
```
Streamlit Cloud pulls LFS files from the repo. Note GitHub's free LFS quota (1 GB storage / 1 GB bandwidth per month).

**Option B: Host the model elsewhere**
Upload `fruits_classification_model.keras` to Hugging Face Hub, a GitHub Release, or similar, and copy its **direct download URL**. Add `*.keras` to `.gitignore`, then in your app's **Settings → Secrets** add:
```toml
MODEL_URL = "https://your-direct-download-link/fruits_classification_model.keras"
```
The app downloads it on first start and caches it. (Google Drive share links are not direct downloads and usually fail.)

**Deploy steps:** push to GitHub → share.streamlit.io → *New app* → select repo, branch, `app.py` → choose Python 3.11 or 3.12 under advanced settings → Deploy.

Also consider: the free tier has limited memory, and the first load takes a while because TensorFlow is large.

## Tech stack
Python · TensorFlow/Keras · NumPy · Pillow · Pandas · Streamlit
