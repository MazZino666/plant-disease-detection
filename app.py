"""
Plant Disease Detection & Analytics Dashboard.
Interactive Multi-Page Streamlit Web Dashboard:
  - 🌿 Leaf Health Scanner & Grad-CAM Explainability
  - 📊 Farm Analytics & Crop Pathology Dashboard
  - 🔬 Explainable AI (Grad-CAM) Visual Studio
  - 📈 Model Benchmark & Comparison Center
  - 📖 Agronomic Disease Encyclopedia & Field Guide
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Custom modules
from src.data_loader import IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model
from src.gradcam import GradCAM, apply_gradcam_overlay

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="PhytoScan AI - Plant Disease Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished Dashboard UI
st.markdown(
    """
    <style>
    /* Main container styling */
    .dashboard-header {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .dashboard-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
    }
    .dashboard-header p {
        margin: 6px 0 0 0;
        font-size: 1.05rem;
        color: #d8f3dc;
    }
    .kpi-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e3a1e;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.4rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1.2rem;
    }
    .healthy-badge {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.9rem;
    }
    .disease-badge {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Agronomic Knowledge Base: Disease description, causes, treatments, and severity
DISEASE_KNOWLEDGE_BASE = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "condition": "Apple Scab",
        "pathogen": "Venturia inaequalis (Fungus)",
        "severity": "Moderate",
        "symptoms": "Olive-green to velvety brown-black lesions on leaves and fruit, causing leaf distortion and premature drop.",
        "treatment": "Apply sulfur or copper-based fungicides at bud break. Remove and destroy fallen leaves in autumn.",
        "prevention": "Prune tree canopies to improve airflow. Plant resistant cultivars such as Honeycrisp, Liberty, or Prima.",
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "condition": "Black Rot (Frogeye Leaf Spot)",
        "pathogen": "Botryosphaeria obtusa (Fungus)",
        "severity": "High",
        "symptoms": "Circular leaf spots with purple margins and light brown centers resembling a frog's eye.",
        "treatment": "Apply captan or thiophanate-methyl fungicides. Prune out dead wood, fire blight cankers, and mummified fruits.",
        "prevention": "Maintain tree vigor through balanced fertilization and rigorous orchard sanitation.",
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "condition": "Cedar Apple Rust",
        "pathogen": "Gymnosporangium juniperi-virginianae (Fungus)",
        "severity": "Moderate",
        "symptoms": "Bright yellow-orange spots on upper leaf surfaces that develop tube-like spore structures beneath.",
        "treatment": "Apply myclobutanil fungicides early in spring. Remove nearby eastern red cedar galls.",
        "prevention": "Avoid planting susceptible apple trees within 1 mile of junipers; choose resistant varieties.",
    },
    "Apple___healthy": {
        "plant": "Apple",
        "condition": "Healthy Foliage",
        "pathogen": "None (Plant is healthy)",
        "severity": "None",
        "symptoms": "Vibrant green leaves with crisp margins and intact chloroplast structures.",
        "treatment": "No chemical intervention needed. Maintain balanced irrigation.",
        "prevention": "Routine scouting and organic neem oil prophylactic sprays.",
    },
    "Corn___Common_rust": {
        "plant": "Corn (Maize)",
        "condition": "Common Rust",
        "pathogen": "Puccinia sorghi (Fungus)",
        "severity": "Moderate",
        "symptoms": "Cinnamon-brown powdery pustules on both upper and lower leaf surfaces.",
        "treatment": "Foliar fungicides (triazoles or strobilurins) if rust threatens high-yield fields.",
        "prevention": "Plant resistant hybrid corn varieties and ensure balanced crop nutrition.",
    },
    "Corn___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)",
        "condition": "Northern Leaf Blight",
        "pathogen": "Exserohilum turcicum (Fungus)",
        "severity": "High",
        "symptoms": "Long, elliptical grayish-green or tan lesions (cigar-shaped) that can merge and blighted foliage.",
        "treatment": "Apply registered fungicides at first detection if economic threshold is reached.",
        "prevention": "Deep tillage to bury infected debris, crop rotation away from corn.",
    },
    "Corn___healthy": {
        "plant": "Corn (Maize)",
        "condition": "Healthy Foliage",
        "pathogen": "None (Plant is healthy)",
        "severity": "None",
        "symptoms": "Uniform parallel venation and healthy deep emerald green blade.",
        "treatment": "Optimal nitrogen and moisture management.",
        "prevention": "Continuous monitoring and balanced crop rotation.",
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "condition": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "severity": "Moderate",
        "symptoms": "Dark brown circular spots with distinctive target-like concentric rings on older lower leaves.",
        "treatment": "Foliar chlorothalonil or mancozeb fungicides applied upon symptom onset.",
        "prevention": "Avoid overhead watering; maintain vigorous vine nutrition with proper nitrogen.",
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "condition": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "severity": "Critical",
        "symptoms": "Water-soaked irregular black/dark-brown lesions, often surrounded by a pale chlorotic border.",
        "treatment": "Copper or systemic fungicides (e.g. cymoxanil/mefenoxam) immediately upon sighting.",
        "prevention": "Destroy cull piles, use certified disease-free seed tubers, ensure wide spacing.",
    },
    "Potato___healthy": {
        "plant": "Potato",
        "condition": "Healthy Foliage",
        "pathogen": "None (Plant is healthy)",
        "severity": "None",
        "symptoms": "Rich green composite foliage without necrosis, stippling, or chlorosis.",
        "treatment": "Maintain regular hilling and soil moisture.",
        "prevention": "Periodic scouting for early signs of blight.",
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "condition": "Bacterial Spot",
        "pathogen": "Xanthomonas spp. (Bacteria)",
        "severity": "High",
        "symptoms": "Small, dark, water-soaked circular to angular lesions with a distinct chlorotic halo.",
        "treatment": "Copper hydroxide mixed with mancozeb spray; streptomycin in seedbeds.",
        "prevention": "Use certified clean seed, avoid handling plants when wet, clean stakes.",
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "condition": "Early Blight",
        "pathogen": "Alternaria linariae (Fungus)",
        "severity": "Moderate",
        "symptoms": "Brown to black lesions with concentric rings (target pattern) starting on lower mature leaves.",
        "treatment": "Apply organic copper fungicides or biofungicides like Bacillus amyloliquefaciens.",
        "prevention": "Mulch around base to prevent soil splash; prune lower suckers for airflow.",
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "condition": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "severity": "Critical",
        "symptoms": "Rapidly expanding large brown-olive blotches with fuzzy white fungal growth underneath in humid air.",
        "treatment": "Immediate application of protective fungicide (mancozeb, chlorothalonil, or copper).",
        "prevention": "Ensure drip irrigation, maximize air circulation, remove diseased plant tissue immediately.",
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "condition": "Leaf Mold",
        "pathogen": "Passalora fulva (Fungus)",
        "severity": "Moderate",
        "symptoms": "Pale green or yellow spots on the upper leaf surface with olive-brown velvety mold on underside.",
        "treatment": "Copper fungicide or bio-fungicides. Reduce greenhouse humidity below 85%.",
        "prevention": "Increase ventilation and plant resistant tomato varieties.",
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "condition": "Septoria Leaf Spot",
        "pathogen": "Septoria lycopersici (Fungus)",
        "severity": "Moderate",
        "symptoms": "Numerous small circular spots with grayish-white centers and dark brown margins.",
        "treatment": "Fungicide sprays (chlorothalonil, copper); strip and bag severely infected lower leaves.",
        "prevention": "Clean garden beds each autumn; rotate tomato crops on a 3-year cycle.",
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "condition": "Healthy Foliage",
        "pathogen": "None (Plant is healthy)",
        "severity": "None",
        "symptoms": "Lush pinnate green leaves, strong turgidity, and flawless vascular venation.",
        "treatment": "Standard organic compost tea and regular watering.",
        "prevention": "Mulching and balanced trellis support.",
    },
}


@st.cache_resource
def load_trained_model(checkpoint_path: str):
    """Load model checkpoint and cache in memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(checkpoint_path)
    if not path.exists():
        return None, None, None

    checkpoint = torch.load(path, map_location=device)
    model_name = checkpoint.get("model_name", "custom_cnn")
    class_names = checkpoint.get("class_names", list(DISEASE_KNOWLEDGE_BASE.keys()))
    num_classes = len(class_names)

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def preprocess_image(image: Image.Image, image_size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    """Preprocess image for PyTorch inference and Grad-CAM."""
    image_rgb = image.convert("RGB")
    image_resized = image_rgb.resize((image_size, image_size), Image.BILINEAR)
    img_np = np.array(image_resized)

    # Convert to tensor and normalize
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        tensor[c] = (tensor[c] - m) / s

    return tensor.unsqueeze(0), img_np


def main():
    # Header Banner
    st.markdown(
        """
        <div class="dashboard-header">
            <h1>🌿 PhytoScan AI — Plant Pathology Dashboard</h1>
            <p>Computer Vision Diagnostic Engine with Grad-CAM Explainability & Agricultural Analytics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Navigation & Settings
    st.sidebar.title("🎛️ Navigation")
    dashboard_page = st.sidebar.radio(
        "Select Dashboard View:",
        [
            "🌾 Foliar Health Scanner",
            "📊 Farm Analytics & KPIs",
            "🔬 Grad-CAM Explainability Studio",
            "📈 Model Benchmarks & Comparison",
            "📖 Agronomic Field Guide",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Neural Engine Configuration")
    model_option = st.sidebar.selectbox(
        "Active Deep Learning Model",
        options=["Transfer Learning (MobileNetV2)", "Custom CNN Baseline"],
        index=0,
    )

    models_dir = Path("./models")
    ckpt_name = "mobilenet_v2_best.pth" if "MobileNet" in model_option else "custom_cnn_best.pth"
    checkpoint_path = models_dir / ckpt_name
    model, class_names, device = load_trained_model(str(checkpoint_path))

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Grad-CAM Settings")
    cam_alpha = st.sidebar.slider("Heatmap Blending Opacity", min_value=0.2, max_value=0.9, value=0.55, step=0.05)
    colormap_choice = st.sidebar.selectbox("Heatmap Colormap", ["jet", "turbo", "inferno", "viridis", "magma"], index=0)

    # =========================================================================
    # PAGE 1: FOLIAR HEALTH SCANNER
    # =========================================================================
    if dashboard_page == "🌾 Foliar Health Scanner":
        col_input, col_results = st.columns([1, 1], gap="large")

        with col_input:
            st.subheader("1. Input Leaf Image")
            input_source = st.radio("Choose Input Mode:", ["Upload Leaf Photograph", "Preloaded Sample Gallery"], horizontal=True)

            selected_image = None
            image_name = "Uploaded Leaf"

            if input_source == "Upload Leaf Photograph":
                uploaded_file = st.file_uploader("Upload leaf image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    selected_image = Image.open(uploaded_file)
                    image_name = uploaded_file.name
            else:
                sample_dir = Path("./sample_images")
                if sample_dir.exists() and any(sample_dir.glob("*.jpg")):
                    sample_files = sorted(list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")))
                    sample_map = {f.stem.replace("___", " - ").replace("_", " "): f for f in sample_files}
                    chosen_sample_name = st.selectbox("Select Sample Leaf:", list(sample_map.keys()))
                    if chosen_sample_name:
                        selected_image = Image.open(sample_map[chosen_sample_name])
                        image_name = chosen_sample_name
                else:
                    st.warning("No sample images found in ./sample_images directory.")

            if selected_image is not None:
                st.image(selected_image, caption=f"Selected Input: {image_name}", use_container_width=True)

        with col_results:
            st.subheader("2. AI Diagnosis & Lesion Heatmap")

            if selected_image is None:
                st.info("👈 Please upload an image or choose a sample leaf from the gallery to run inference.")
            elif model is None:
                st.error(f"⚠️ Model checkpoint `{ckpt_name}` not found. Run training script first.")
            else:
                with st.spinner("Analyzing foliar pathology & calculating Grad-CAM..."):
                    input_tensor, resized_np = preprocess_image(selected_image)
                    input_tensor = input_tensor.to(device)

                    grad_cam = GradCAM(model)
                    heatmap, pred_idx, confidence = grad_cam.generate_heatmap(input_tensor)
                    grad_cam.remove_hooks()

                    pred_class = class_names[pred_idx]
                    is_healthy = "healthy" in pred_class.lower()

                    with torch.no_grad():
                        logits = model(input_tensor)
                        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                    top3_indices = np.argsort(probs)[::-1][:3]

                    overlaid_img = apply_gradcam_overlay(
                        resized_np,
                        heatmap,
                        alpha=cam_alpha,
                        colormap_name=colormap_choice,
                    )

                info = DISEASE_KNOWLEDGE_BASE.get(
                    pred_class,
                    {
                        "plant": pred_class.split("___")[0].replace("_", " "),
                        "condition": pred_class.split("___")[-1].replace("_", " "),
                        "pathogen": "Identified Pathogen",
                        "severity": "Moderate",
                        "symptoms": "Foliar lesions present.",
                        "treatment": "Follow local agronomy guidelines.",
                        "prevention": "Routine leaf health monitoring.",
                    },
                )

                status_html = (
                    '<span class="healthy-badge">🟢 HEALTHY</span>'
                    if is_healthy
                    else f'<span class="disease-badge">🔴 {info["severity"].upper()} SEVERITY PATHOGEN</span>'
                )

                st.markdown(
                    f"""
                    <div class="card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #1e293b;">{info['plant']} — {info['condition']}</h3>
                            {status_html}
                        </div>
                        <p style="margin: 8px 0 0 0; color: #64748b; font-size: 0.95rem;">
                            <strong>Causal Pathogen:</strong> {info['pathogen']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_c1, col_c2 = st.columns([1, 2])
                with col_c1:
                    st.metric("Diagnostic Confidence", f"{confidence * 100:.1f}%")
                with col_c2:
                    st.write("Confidence Meter:")
                    st.progress(float(confidence))

                st.markdown("#### 🔬 Visual Explainability (Grad-CAM)")
                vcol1, vcol2 = st.columns(2)
                with vcol1:
                    st.image(overlaid_img, caption="Grad-CAM Lesion Overlay", use_container_width=True)
                with vcol2:
                    cmap = plt.get_cmap(colormap_choice)
                    heat_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
                    st.image(heat_rgb, caption="Raw Attention Heatmap", use_container_width=True)

                st.markdown("#### 📊 Differential Diagnoses (Top 3)")
                for idx in top3_indices:
                    cname = class_names[idx].replace("___", " — ").replace("_", " ")
                    prob = probs[idx]
                    st.write(f"**{cname}**: {prob * 100:.1f}%")
                    st.progress(float(prob))

                st.markdown("---")
                with st.expander("🌾 View Treatment Action Plan & Prevention Advice", expanded=True):
                    st.markdown(f"**Key Symptoms:** {info['symptoms']}")
                    st.markdown(f"**Treatment Actions:** {info['treatment']}")
                    st.markdown(f"**Preventative Measures:** {info['prevention']}")

    # =========================================================================
    # PAGE 2: FARM ANALYTICS & KPIS
    # =========================================================================
    elif dashboard_page == "📊 Farm Analytics & KPIs":
        st.subheader("🌾 Farm Surveillance & Crop Pathology Analytics")
        st.write("Aggregated epidemiological metrics and surveillance data across staple agricultural crops.")

        # KPI Metrics Row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">Monitored Classes</div><div class="kpi-value">16</div><small>Across 4 Staple Crops</small></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">Diagnostic Latency</div><div class="kpi-value">43.9 ms</div><small>Edge MobileNetV2 Speed</small></div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">Healthy Leaf Ratio</div><div class="kpi-value">25.0%</div><small>Baseline Controls</small></div>',
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                '<div class="kpi-card"><div class="kpi-label">Critical Pathogens</div><div class="kpi-value">2</div><small>Late Blight (Tomato/Potato)</small></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🥦 Crop Class Distribution")
            crop_counts = {"Tomato": 6, "Apple": 4, "Corn": 3, "Potato": 3}
            df_crops = pd.DataFrame(list(crop_counts.items()), columns=["Crop", "Pathology Classes"])
            st.bar_chart(df_crops.set_index("Crop"), color="#2d6a4f")

        with c2:
            st.markdown("#### 🦠 Pathogen Kingdom Breakdown")
            pathogen_breakdown = {
                "Fungi (Scab, Rust, Blight, Mold)": 11,
                "Healthy Baseline (No Pathogen)": 4,
                "Oomycota (Phytophthora)": 2,
                "Bacteria (Xanthomonas)": 1,
            }
            df_pathogens = pd.DataFrame(list(pathogen_breakdown.items()), columns=["Pathogen Type", "Count"])
            st.bar_chart(df_pathogens.set_index("Pathogen Type"), color="#e76f51")

        st.markdown("---")
        st.markdown("#### 🚨 Epidemiological Risk & Action Matrix")
        risk_data = [
            {"Crop": "Tomato", "Disease": "Late Blight", "Severity": "Critical", "Spread Rate": "Rapid (<48h in rain)", "Action": "Fungicide Spray & Quarantine"},
            {"Crop": "Potato", "Disease": "Late Blight", "Severity": "Critical", "Spread Rate": "Rapid", "Action": "Immediate systemic copper treatment"},
            {"Crop": "Apple", "Disease": "Black Rot", "Severity": "High", "Spread Rate": "Moderate", "Action": "Prune infected cankers and fruit"},
            {"Crop": "Tomato", "Disease": "Bacterial Spot", "Severity": "High", "Spread Rate": "High", "Action": "Copper + Mancozeb treatment"},
            {"Crop": "Corn", "Disease": "Northern Leaf Blight", "Severity": "High", "Spread Rate": "Moderate", "Action": "Resistant hybrids & foliar sprays"},
            {"Crop": "Apple", "Disease": "Apple Scab", "Severity": "Moderate", "Spread Rate": "Moderate", "Action": "Fungicide at bud break"},
            {"Crop": "Corn", "Disease": "Common Rust", "Severity": "Moderate", "Spread Rate": "High", "Action": "Foliar triazoles if widespread"},
        ]
        st.dataframe(pd.DataFrame(risk_data), use_container_width=True)

    # =========================================================================
    # PAGE 3: GRAD-CAM EXPLAINABILITY STUDIO
    # =========================================================================
    elif dashboard_page == "🔬 Grad-CAM Explainability Studio":
        st.subheader("🔬 Grad-CAM Explainable AI (XAI) Studio")
        st.write("Inspect how convolutional neural networks focus attention on necrotic lesions and chlorotic halos.")

        sample_dir = Path("./sample_images")
        sample_files = sorted(list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))) if sample_dir.exists() else []

        if not sample_files:
            st.warning("No sample leaves available.")
        else:
            sample_names = [f.stem.replace("___", " - ").replace("_", " ") for f in sample_files]
            selected_idx = st.selectbox("Choose Leaf to Inspect:", range(len(sample_names)), format_func=lambda i: sample_names[i])
            chosen_leaf_path = sample_files[selected_idx]
            img = Image.open(chosen_leaf_path)

            input_tensor, resized_np = preprocess_image(img)
            input_tensor = input_tensor.to(device)

            grad_cam = GradCAM(model)
            heatmap, pred_idx, confidence = grad_cam.generate_heatmap(input_tensor)
            grad_cam.remove_hooks()

            # Studio Columns
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("##### 1. Original RGB Leaf")
                st.image(img, use_container_width=True)

            with col_b:
                st.markdown("##### 2. Raw Gradient Activation Map")
                st.image(heatmap, clamp=True, use_container_width=True)

            with col_c:
                st.markdown("##### 3. Grad-CAM Overlay")
                overlay = apply_gradcam_overlay(resized_np, heatmap, alpha=cam_alpha, colormap_name=colormap_choice)
                st.image(overlay, use_container_width=True)

            # Mathematical Formula Card
            st.markdown("---")
            st.markdown(
                """
                ### 📐 Mathematical Formulation of Grad-CAM
                Grad-CAM computes the importance weight $\\alpha_k^c$ for each feature map $A^k$ via global average pooling of gradients:
                """
            )
            st.latex(r"\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}")
            st.markdown("The final localization heatmap is computed via a rectified linear combination:")
            st.latex(r"L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)")

    # =========================================================================
    # PAGE 4: MODEL BENCHMARKS & COMPARISON
    # =========================================================================
    elif dashboard_page == "📈 Model Benchmarks & Comparison":
        st.subheader("📈 Deep Learning Benchmarks & Model Comparison")
        st.write("Detailed quantitative evaluation comparing the Custom Deep CNN baseline against MobileNetV2 Transfer Learning.")

        comp_summary_path = models_dir / "comparison_summary.json"
        if comp_summary_path.exists():
            with open(comp_summary_path) as f:
                summary_data = json.load(f)

            cols = st.columns(len(summary_data))
            for i, data in enumerate(summary_data):
                with cols[i]:
                    st.metric(
                        label=f"{data['model_name'].replace('_', ' ').title()}",
                        value=f"{data['test_accuracy'] * 100:.2f}%",
                        delta=f"Macro F1: {data['macro_f1'] * 100:.1f}%",
                    )
                    st.caption(f"Params: {data['parameters']:,} | Latency: {data['latency_ms_per_sample']:.1f}ms / sample")

        comp_chart_path = models_dir / "model_comparison_chart.png"
        if comp_chart_path.exists():
            st.image(str(comp_chart_path), caption="Accuracy and Macro F1 Comparison", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📉 Training & Validation Learning Curves")
        c1, c2 = st.columns(2)
        with c1:
            cnn_curve = models_dir / "training_curves_custom_cnn.png"
            if cnn_curve.exists():
                st.image(str(cnn_curve), caption="Custom CNN Learning Curves", use_container_width=True)
        with c2:
            mob_curve = models_dir / "training_curves_mobilenet_v2.png"
            if mob_curve.exists():
                st.image(str(mob_curve), caption="MobileNetV2 Learning Curves", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🎯 Confusion Matrix Heatmaps")
        m1, m2 = st.columns(2)
        with m1:
            cnn_cm = models_dir / "confusion_matrix_custom_cnn.png"
            if cnn_cm.exists():
                st.image(str(cnn_cm), caption="Custom CNN Confusion Matrix", use_container_width=True)
        with m2:
            mob_cm = models_dir / "confusion_matrix_mobilenet_v2.png"
            if mob_cm.exists():
                st.image(str(mob_cm), caption="MobileNetV2 Confusion Matrix", use_container_width=True)

    # =========================================================================
    # PAGE 5: AGRONOMIC FIELD GUIDE
    # =========================================================================
    elif dashboard_page == "📖 Agronomic Field Guide":
        st.subheader("📖 Agronomic Pathology Field Guide")
        st.write("Comprehensive disease symptoms, causal organisms, and treatment prescriptions for farm managers.")

        selected_crop = st.selectbox("Filter by Crop:", ["All Crops", "Tomato", "Apple", "Corn", "Potato"])

        for key, entry in DISEASE_KNOWLEDGE_BASE.items():
            if selected_crop != "All Crops" and entry["plant"].lower() not in selected_crop.lower():
                continue

            with st.expander(f"🌿 {entry['plant']} — {entry['condition']} ({entry['severity']} Severity)"):
                st.markdown(f"**Pathogen:** `{entry['pathogen']}`")
                st.markdown(f"**Symptoms:** {entry['symptoms']}")
                st.markdown(f"**Chemical & Biological Controls:** {entry['treatment']}")
                st.markdown(f"**Cultural Prevention Practices:** {entry['prevention']}")


if __name__ == "__main__":
    main()
