# 🌿 Plant Disease Detection using Deep Learning (CNN & Transfer Learning) with Grad-CAM Explainability

An end-to-end computer vision and deep learning system that classifies foliar plant diseases and healthy leaves across staple crops (Tomato, Potato, Apple, Corn), provides model interpretability through **Grad-CAM (Gradient-weighted Class Activation Mapping)**, and features an interactive **Streamlit web application** for real-time agricultural diagnostics.

---

## 📌 Project Overview & Highlights

- **Dual Neural Network Architectures:**
  - **Custom CNN Baseline:** 4 convolutional blocks with `Conv2D`, `BatchNorm2d`, `ReLU`, `MaxPool2d`, `Dropout2d`, and dense classifier.
  - **Transfer Learning Backbone:** Pre-trained **MobileNetV2** (lightweight edge-optimized) and **ResNet50** with custom classification heads.
- **Data Augmentation & Robust Preprocessing:**
  - Resize to $224 \times 224$ pixels, ImageNet normalization.
  - Real-time data augmentation: random rotations ($\pm 30^\circ$), horizontal and vertical flips, random zoom/crop, and color jitter.
- **Modern Training Engine:**
  - Early stopping with best weight restoration.
  - Model checkpointing (`best_model.pth` and `last_model.pth`).
  - Dynamic learning rate scheduling via `ReduceLROnPlateau`.
- **Comprehensive Evaluation & Benchmarking:**
  - Accuracy, Precision, Recall, Macro & Weighted F1-scores.
  - High-resolution Seaborn confusion matrix heatmaps.
  - Side-by-side comparative analysis (Custom CNN vs. Transfer Learning).
- **Explainable AI (XAI) with Grad-CAM:**
  - Generates spatial activation maps showing exactly which leaf lesions and necrotic spots influenced the neural network's diagnosis.
- **Interactive Streamlit Web App (`app.py`):**
  - Instant leaf photo upload or preloaded sample gallery.
  - Real-time prediction, confidence score meter, and top-3 differential diagnoses.
  - Grad-CAM heatmap visualization with opacity and colormap sliders.
  - Agronomic guidance: symptoms, biological/chemical treatments, and preventive farming practices.

---

## 🗂️ Project Directory Structure

```text
plant-disease-detection/
├── data/
│   ├── download_dataset.py        # Dataset downloader, organizer (train/val/test), and generator
│   ├── train/                     # 70% Training images organized by class
│   ├── val/                       # 15% Validation images
│   └── test/                      # 15% Held-out test images
├── models/                        # Checkpoints, learning curves, confusion matrices, comparison logs
│   ├── custom_cnn_best.pth
│   ├── mobilenet_v2_best.pth
│   ├── training_curves_*.png
│   ├── confusion_matrix_*.png
│   └── model_comparison_chart.png
├── notebooks/
│   └── 01_eda_and_experiments.ipynb # Exploratory Data Analysis & visual experiment workflow
├── sample_images/                 # Preloaded leaf images for instant Streamlit testing
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Transforms, augmentation, and PyTorch DataLoaders
│   ├── model.py                   # Custom CNN, MobileNetV2, and ResNet50 model definitions
│   ├── gradcam.py                 # Grad-CAM hooks, heatmap generation, and image overlay
│   ├── train.py                   # Training loop with early stopping, checkpointing, and scheduler
│   └── evaluate.py                # Test set evaluation, metrics calculation, and model comparison
├── app.py                         # Streamlit interactive web diagnosis dashboard
├── requirements.txt               # Project dependencies
└── README.md                      # Comprehensive project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open the Workspace
```bash
cd "C:\Users\shiva\.gemini\antigravity\scratch\plant-disease-detection"
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation

The project is structured around the **PlantVillage** dataset, containing healthy leaves and diseased classes:
- **Apple:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- **Corn (Maize):** Common Rust, Northern Leaf Blight, Healthy
- **Potato:** Early Blight, Late Blight, Healthy
- **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Healthy

### Option A: Automatic Dataset Setup & Built-in Generator
Run the dataset organizer to automatically generate a balanced dataset split into `train/` (70%), `val/` (15%), and `test/` (15%), as well as demo leaves in `sample_images/`:
```bash
python data/download_dataset.py --samples_per_class 45
```

### Option B: Using an Existing PlantVillage Directory
If you downloaded PlantVillage from Kaggle or tensorflow_datasets, split and organize it automatically:
```bash
python data/download_dataset.py --source_dir /path/to/raw/PlantVillage
```

---

## 🚀 Model Training

### 1. Train the Custom CNN Baseline
```bash
python src/train.py --model custom_cnn --epochs 12 --batch_size 32 --lr 0.001 --patience 4
```

### 2. Train the MobileNetV2 Transfer Learning Model
```bash
python src/train.py --model mobilenet_v2 --epochs 12 --batch_size 32 --lr 0.0005 --patience 4
```

During training, the system outputs:
- Real-time epoch loss and accuracy for train and validation sets.
- Model checkpointing whenever validation loss achieves a new minimum.
- Early stopping if no improvement is observed for 4 consecutive epochs.
- Saves learning curves to `models/training_curves_<model_name>.png`.

---

## 📊 Evaluation & Model Comparison

Evaluate any saved checkpoint on the held-out test set:
```bash
python src/evaluate.py --checkpoint models/custom_cnn_best.pth
python src/evaluate.py --checkpoint models/mobilenet_v2_best.pth
```

### Compare Both Models:
```bash
python src/evaluate.py --compare models/custom_cnn_best.pth models/mobilenet_v2_best.pth
```
This produces:
1. `models/confusion_matrix_*.png`: Confusion matrix heatmaps.
2. `models/metrics_*.json`: Full precision, recall, and F1-score breakdowns.
3. `models/model_comparison_chart.png`: Side-by-side bar chart of accuracy and F1-score.

---

## 🔬 Grad-CAM (Visual Explainability)

Grad-CAM highlights the exact regions in the input image that were critical for the model's decision:
$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
where $\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{i,j}}$.

You can test Grad-CAM via Python:
```python
from PIL import Image
from src.gradcam import GradCAM, apply_gradcam_overlay
from src.model import build_model
from src.data_loader import get_transforms

model = build_model("mobilenet_v2", num_classes=16)
# Load weights, feed image tensor, compute heatmap, and overlay
```

---

## 🌐 Running the Streamlit Web Application

Launch the interactive web UI:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
Features:
- **Model Switcher:** Toggle between Custom CNN and MobileNetV2.
- **Image Input:** Drag and drop any leaf photograph or select from the preloaded sample gallery.
- **Diagnosis Card:** Instant disease status (Healthy vs. Diseased), pathogen identification, confidence meter, and top-3 differential diagnoses.
- **Grad-CAM Heatmap Viewer:** Adjust blending opacity and colormaps in real-time.
- **Agronomic Guidance:** Practical management advice and prevention tips for growers.
- **Performance Tab:** View training curves and confusion matrix benchmarks.

---

## 📜 License
This project is open-source under the MIT License.
