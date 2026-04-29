# src/infer_single.py (uses saved threshold)
import sys, os
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess
from model_improved import build_siamese

MODEL_PATH = r"E:\handwriting_matcher\outputs\checkpoints\siamese_best.h5"
THRESH_PATH = r"E:\handwriting_matcher\outputs\threshold.txt"

if len(sys.argv) < 3:
    print("Usage: python src\\infer_single.py <imageA> <imageB>")
    sys.exit(1)

a_path = sys.argv[1]
b_path = sys.argv[2]

# rebuild model and load weights
model = build_siamese()
try:
    model.load_weights(MODEL_PATH)
except Exception as e:
    # fallback: try load_model
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH, compile=False)

# read threshold (default 0.5)
thr = 0.5
if os.path.exists(THRESH_PATH):
    try:
        thr = float(open(THRESH_PATH).read().strip())
    except:
        pass

A = np.expand_dims(load_and_preprocess(a_path),0)
B = np.expand_dims(load_and_preprocess(b_path),0)
score = float(model.predict([A,B], verbose=0)[0][0])
print(f"Similarity score (0-1): {score:.4f}")
print(f"Using threshold: {thr:.4f}")
print("PREDICTION:", "MATCH" if score >= thr else "NOT MATCH")
