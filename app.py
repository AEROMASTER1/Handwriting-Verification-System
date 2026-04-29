# app/app.py
import sys
sys.path.append(r"E:\handwriting_matcher\src")   # allow imports from src/

import streamlit as st
import numpy as np
import tempfile
import os
from preprocess import load_and_preprocess
from model_improved import build_siamese
from extract_features import extract_handwriting_features
import joblib

# Paths - change if you saved models under different names
SIAMESE_WEIGHTS = r"E:\handwriting_matcher\outputs\checkpoints\siamese_best.h5"
FUSION_MODEL_PATH = r"E:\handwriting_matcher\outputs\fusion_model.pkl"

# Page config
st.set_page_config(page_title="Hand Writing Matcher", layout="wide")

# Title (centered)
st.markdown(
    "<h1 style='text-align:center; color:#0B5FFF; font-weight:700;'>Hand Writing Matcher</h1>",
    unsafe_allow_html=True
)

# Sidebar controls (sliders) - with requested default values
st.sidebar.header("Tuning (live)")
CNN_WEIGHT = st.sidebar.slider("CNN weight", 0.0, 1.0, 0.30, 0.05)
FUSION_WEIGHT = st.sidebar.slider("Fusion weight", 0.0, 1.0, 0.70, 0.05)
FINAL_THRESHOLD = st.sidebar.slider("Final threshold", 0.0, 1.0, 0.59, 0.01)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#6c757d'>Tips: Increase CNN weight if visual similarity should dominate. Raise threshold to reduce false MATCHes.</small>",
    unsafe_allow_html=True
)

# Input area: two uploaders with simple labels (no 'Preview' word)
col1, col2, col3 = st.columns([1,1,0.4])
with col1:
    st.subheader("Image A")
    img1 = st.file_uploader(" ", key="imgA", type=['png','jpg','jpeg','tif','tiff','bmp'], accept_multiple_files=False)
    if img1:
        st.image(img1, use_container_width=True)

with col2:
    st.subheader("Image B")
    img2 = st.file_uploader(" ", key="imgB", type=['png','jpg','jpeg','tif','tiff','bmp'], accept_multiple_files=False)
    if img2:
        st.image(img2, use_container_width=True)

with col3:
    st.markdown("### Actions")
    st.write("Upload two images and press **Check Match**.")
    check_btn = st.button("Check Match", use_container_width=True)

# Lazy loading models (cached)
@st.cache_resource
def load_models():
    siam = build_siamese()
    try:
        siam.load_weights(SIAMESE_WEIGHTS)
    except Exception:
        from tensorflow.keras.models import load_model
        siam = load_model(SIAMESE_WEIGHTS, compile=False)
    clf = joblib.load(FUSION_MODEL_PATH)
    return siam, clf

# Utility to safely get fusion probability for class 1
def get_fusion_prob(clf, fused_vector):
    classes = getattr(clf, "classes_", None)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(fused_vector)[0]
        try:
            match_idx = list(classes).index(1)
        except Exception:
            match_idx = len(probs) - 1
        return float(probs[match_idx])
    else:
        p = int(clf.predict(fused_vector)[0])
        return 1.0 if p == 1 else 0.0

# Run prediction when button pressed
if check_btn:
    if not img1 or not img2:
        st.error("Please upload both images before checking.")
    else:
        # Save uploaded files temporarily (Windows-safe)
        t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            t1.write(img1.getvalue()); t2.write(img2.getvalue())
            t1.flush(); t2.flush()
            t1.close(); t2.close()

            # load models
            siamese, fusion_clf = load_models()

            # preprocess images for cnn (ensure same settings as during training)
            A = np.expand_dims(load_and_preprocess(t1.name), 0)
            B = np.expand_dims(load_and_preprocess(t2.name), 0)

            # CNN similarity score
            cnn_score = float(siamese.predict([A, B], verbose=0)[0][0])

            # handcrafted features
            featA = extract_handwriting_features(t1.name)
            featB = extract_handwriting_features(t2.name)
            diff = np.abs(np.array(featA) - np.array(featB))

            # fused vector same as during training
            fused = np.concatenate(([cnn_score], diff)).reshape(1, -1)

            # fusion probability
            fusion_prob = get_fusion_prob(fusion_clf, fused)

            # combined final score
            final_score = float(CNN_WEIGHT * cnn_score + FUSION_WEIGHT * fusion_prob)

            # decision
            is_match = final_score >= FINAL_THRESHOLD

            # Display results with color accents
            res_col1, res_col2 = st.columns([1,1])
            with res_col1:
                st.markdown("### Result")
                if is_match:
                    st.markdown(
                        f"<div style='background:#E8F8F5;padding:10px;border-radius:8px'>"
                        f"<h2 style='color:#0F9D58'>✅ MATCH</h2>"
                        f"<p style='font-size:16px;color:#264653'>Likely the same writer</p>"
                        f"</div>", unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background:#FFF5F5;padding:10px;border-radius:8px'>"
                        f"<h2 style='color:#D00000'>❌ NOT MATCH</h2>"
                        f"<p style='font-size:16px;color:#6b2b2b'>Likely different writers</p>"
                        f"</div>", unsafe_allow_html=True
                    )

                st.markdown(f"<b>Combined score:</b> <span style='color:#0B5FFF'>{final_score:.4f}</span> (threshold = {FINAL_THRESHOLD:.2f})", unsafe_allow_html=True)
                st.markdown(f"<b>CNN similarity:</b> <span style='color:#0B5FFF'>{cnn_score:.4f}</span>", unsafe_allow_html=True)
                st.markdown(f"<b>Fusion probability:</b> <span style='color:#0B5FFF'>{fusion_prob:.4f}</span>", unsafe_allow_html=True)

            with res_col2:
                st.markdown("### Diagnostic")
                st.write("Use the expanders below to inspect features and classifier info.")
                with st.expander("Show handcrafted features & diffs"):
                    st.write("Features A:", np.round(featA, 4).tolist())
                    st.write("Features B:", np.round(featB, 4).tolist())
                    st.write("Absolute diffs:", np.round(diff.tolist(), 4))
                with st.expander("Fusion classifier info"):
                    try:
                        st.write("Classes:", fusion_clf.classes_)
                    except Exception:
                        st.write("Could not read classifier classes_")
                    try:
                        st.write("Classifier type:", type(fusion_clf).__name__)
                    except:
                        pass

        finally:
            # cleanup temp files (best-effort)
            for p in (t1.name, t2.name):
                try:
                    os.remove(p)
                except:
                    pass

# Footer / help (subtle)
st.markdown("---")
st.markdown(
    "<div style='color:#6c757d'>How it works: CNN computes visual similarity, handcrafted features capture handwriting-style differences, and a fusion classifier combines both. Use the sliders in the sidebar to tune behavior for demo.</div>",
    unsafe_allow_html=True
)
