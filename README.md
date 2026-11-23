# 🍎 Freshness AI – Fresh vs Rotten Fruit Detection

<p align="center">

**An AI-powered computer vision application that detects whether a fruit is Fresh or Rotten using a custom Convolutional Neural Network (CNN).**

Built with Python, TensorFlow/Keras, Streamlit, and Hugging Face.

</p>

---

## 🚀 Project Overview

Freshness AI is a deep learning-based image classification project designed to identify whether a fruit appears fresh or rotten.

The project uses a custom-built Convolutional Neural Network (CNN) trained on fruit images. Users can upload an image through an interactive Streamlit web application and receive an instant prediction with confidence scores.

The original six-class dataset was converted into a binary classification problem:

- **Fresh (0):** Fresh apples, bananas, and oranges.
- **Rotten (1):** Rotten apples, bananas, and oranges.

The system focuses exclusively on fruit freshness classification. It does not identify the fruit type.

## ✨ Key Features

- **AI-Powered Detection:** Custom CNN model for Fresh vs Rotten classification.
- **Image Upload:** Supports JPG, JPEG, and PNG images.
- **Real-Time Prediction:** Analyze uploaded images through the Streamlit interface.
- **Confidence Scores:** Displays prediction confidence and both class probabilities.
- **Interactive Dashboard:** Modern interface with sidebar navigation.
- **Prediction History:** Maintains prediction records during the current session.
- **CSV Export:** Download prediction history for further analysis.
- **Model Information:** View CNN architecture and evaluation metrics.
- **Cloud Model Hosting:** Stores the trained model on Hugging Face Hub instead of GitHub.
- **Automatic Model Download:** Downloads the model from Hugging Face when the application loads it.
- **Model Caching:** Uses Streamlit caching to avoid repeatedly loading the model.

## 🎯 Model Architecture

I developed a custom CNN using TensorFlow/Keras.

The model consists of three convolutional blocks followed by fully connected layers and a sigmoid output neuron.

### Architecture

```text
Input Image (224 × 224 × 3)
          |
          v
Conv2D (32 filters, 3×3, ReLU)
          |
MaxPooling2D (2×2)
          |
          v
Conv2D (64 filters, 3×3, ReLU)
          |
MaxPooling2D (2×2)
          |
          v
Conv2D (128 filters, 3×3, ReLU)
          |
MaxPooling2D (2×2)
          |
          v
       Flatten
          |
          v
Dense (128, ReLU)
          |
          v
Dense (1, Sigmoid)
          |
          v
   Fresh / Rotten
```

### Model Configuration

| Parameter | Configuration |
|---|---|
| Model Type | Custom CNN |
| Framework | TensorFlow / Keras |
| Input Shape | 224 × 224 × 3 |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Output Activation | Sigmoid |
| Classification Type | Binary Classification |
| Decision Threshold | 0.5 |
| Training Epochs | 20 |

## 📊 Model Performance

The model was trained for 20 epochs and evaluated on a separate test dataset.

### Evaluation Results

| Metric | Result |
|---|---:|
| Training Accuracy | 97.82% |
| Validation Accuracy | 98.35% |
| Test Accuracy | 98.15% |
| Test Loss | 0.0490 |

These results represent performance on the available dataset splits. They do not guarantee the same accuracy on every real-world image.

## 📁 Dataset Information

The original dataset contains six fruit categories.

### Original Classes

| Original Class | Binary Label | Category |
|---|---:|---|
| freshapples | 0 | Fresh |
| freshbanana | 0 | Fresh |
| freshoranges | 0 | Fresh |
| rottenapples | 1 | Rotten |
| rottenbanana | 1 | Rotten |
| rottenoranges | 1 | Rotten |

### Binary Classification

The six original classes were combined into two categories.

```python
# Fresh = 0
# Rotten = 1

if prediction >= 0.5:
    result = "Rotten"
else:
    result = "Fresh"
```

### Image Preprocessing

The images undergo the following preprocessing steps:

1. Image loading.
2. RGB color conversion.
3. Resizing to 224 × 224 pixels using bilinear interpolation.
4. Conversion into NumPy arrays.
5. Pixel normalization by dividing by 255.0.
6. Batch dimension expansion.

The same input dimensions and normalization are used during application inference.

## ☁️ Model Hosting – Hugging Face

The trained CNN model is approximately **134 MB**, which exceeds GitHub's standard 100 MB individual file limit.

Instead of storing the model directly in GitHub, I hosted it on Hugging Face Hub.

### Hugging Face Repository

**Repository ID:**

```text
vishal953436/fruit-freshness-model
```

**Model File:**

```text
fruits_classification_model.keras
```

**Model Repository:**

https://huggingface.co/vishal953436/fruit-freshness-model

### How Model Downloading Works

The application uses the `huggingface_hub` library to download the trained model automatically.

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="vishal953436/fruit-freshness-model",
    filename="fruits_classification_model.keras",
    repo_type="model"
)
```

TensorFlow then loads the downloaded model:

```python
model = tf.keras.models.load_model(
    model_path,
    compile=False
)
```

### Benefits of Hugging Face Hosting

- Keeps large model files outside the GitHub repository.
- Provides centralized model storage.
- Supports downloading models directly into Python applications.
- Automatically caches downloaded files.
- Makes the application repository smaller and easier to maintain.

The Hugging Face repository is public, so an access token is not required for downloading the model.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Deep learning framework |
| CNN | Image classification |
| NumPy | Numerical operations |
| Pillow | Image processing |
| Pandas | Prediction history and CSV export |
| Streamlit | Web application development |
| Hugging Face Hub | Model hosting and downloading |
| Git & GitHub | Version control and source code hosting |

## 📂 Project Structure

```text
Fruit_Freshness_Detection/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
├── utils/
│   ├── __init__.py
│   └── prediction.py
│
└── assets/
    ├── home_dashboard.png
    ├── fresh_prediction.png
    ├── rotten_prediction.png
    └── prediction_history.png
```

**Note:** The trained `.keras` model is hosted on Hugging Face and is not stored in the GitHub repository.

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project directory:

```bash
cd Fruit_Freshness_Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```powershell
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

Open the local application:

```text
http://localhost:8501
```

### 5. Automatic Model Download

When the application loads the model for the first time:

1. The application connects to Hugging Face Hub.
2. It downloads `fruits_classification_model.keras`.
3. TensorFlow loads the trained CNN.
4. Streamlit caches the loaded model.
5. The model becomes ready for image predictions.

An internet connection is required for the initial model download.

## 🧪 How to Use the Application

1. Launch the Streamlit application.
2. Navigate to **Fruit Detection** using the sidebar.
3. Upload an image of an apple, banana, or orange.
4. Click **Analyze Fruit**.
5. Wait for the CNN model to process the image.
6. View the prediction result.

### Prediction Output

The application displays:

- Predicted class: Fresh or Rotten.
- Prediction confidence.
- Fresh probability.
- Rotten probability.

Example:

```text
Prediction: Fresh

Fresh Probability: 96.5%
Rotten Probability: 3.5%
```

The example above is illustrative, not an actual model prediction.

## 🌍 Real-World Testing

To evaluate the application beyond the original dataset, test it with images captured using a smartphone or camera.

Recommended test scenarios:

- Fresh apples, bananas, and oranges.
- Rotten apples, bananas, and oranges.
- Different lighting conditions.
- Different backgrounds.
- Different camera angles and distances.
- Blurry and low-quality images.
- Images containing objects other than fruit.

Compare predictions against the actual condition of the fruit and record incorrect predictions for future improvements.

## 🚀 Deployment – Streamlit Community Cloud

The application can be deployed using Streamlit Community Cloud.

### Step 1: Push the Application to GitHub

Create a GitHub repository and upload your application source code.

Make sure the `.keras` model is excluded from GitHub using `.gitignore`.

### Step 2: Deploy the Application

1. Visit https://share.streamlit.io/
2. Sign in using your GitHub account.
3. Select **Create app**.
4. Choose your GitHub repository and branch.
5. Set the main file path to `app.py`.
6. Select a Python version compatible with your TensorFlow dependencies.
7. Click **Deploy**.

### Step 3: Model Download During Deployment

When Streamlit Cloud starts the application:

- The application connects to the public Hugging Face repository.
- The trained model is downloaded using `hf_hub_download()`.
- TensorFlow loads the model.
- Streamlit caches the loaded model for subsequent reruns.

Make sure the deployment environment has sufficient memory and storage for TensorFlow and the approximately 134 MB model file.

## ⚠️ Limitations

- The model was trained using images of apples, bananas, and oranges.
- It only predicts Fresh or Rotten; it does not identify fruit type.
- Real-world accuracy may vary with lighting, backgrounds, image quality, and camera conditions.
- The model may produce unreliable predictions for fruits outside its training distribution.
- A high prediction confidence does not guarantee a correct result.
- The system is an image-classification demonstration, not a food-safety inspection tool.

## 🔮 Future Improvements

- [ ] Add Grad-CAM visualizations for model explainability.
- [ ] Expand the dataset with additional fruit categories.
- [ ] Improve performance on real-world images.
- [ ] Add fruit-type classification.
- [ ] Develop a mobile-friendly application.
- [ ] Optimize inference speed and model size.
- [ ] Explore live camera-based fruit freshness detection.

## 👨‍💻 Author

**Vishal Bhatti**

B.Tech – Artificial Intelligence and Robotics Engineering

Interested in Artificial Intelligence, Machine Learning, Deep Learning, and Computer Vision.

### Connect With Me

- GitHub: [Your GitHub Profile](https://github.com/vishalbhatti00795-wq)
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/vishal-bhatti-1a864032b/)

---

<p align="center">

**Built with Python, TensorFlow, Streamlit, and Hugging Face.**

If you find this project interesting, consider giving the repository a ⭐!

</p>
