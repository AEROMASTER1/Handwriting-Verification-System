# src/compute_threshold.py
import csv, os, numpy as np
from sklearn.metrics import f1_score, roc_curve
from preprocess import load_and_preprocess
from model_improved import build_siamese
import tensorflow as tf

PAIRS_CSV = r"E:\handwriting_matcher\pairs\train_pairs.csv"
MODEL_PATH = r"E:\handwriting_matcher\outputs\checkpoints\siamese_best.h5"
OUT_PATH = r"E:\handwriting_matcher\outputs\threshold.txt"

# load pairs
pairs = []
labels = []
with open(PAIRS_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        pairs.append((r['imgA'], r['imgB']))
        labels.append(int(r['label']))
pairs = np.array(pairs)
labels = np.array(labels)

# use last 15% as test (same split used previously)
n = len(labels)
start = int(n * 0.85)
pairs_test = pairs[start:]
labels_test = labels[start:]

print("Test size:", len(labels_test))

# rebuild model and load weights
model = build_siamese()
try:
    model.load_weights(MODEL_PATH)
    print("Weights loaded")
except Exception as e:
    print("Could not load weights:", e)
    raise

# compute scores
scores = []
for a,b in pairs_test:
    A = tf.expand_dims(load_and_preprocess(a),0).numpy()
    B = tf.expand_dims(load_and_preprocess(b),0).numpy()
    s = float(model.predict([A,B], verbose=0)[0][0])
    scores.append(s)
scores = np.array(scores)

# compute ROC thresholds and best F1
fpr, tpr, thresholds = roc_curve(labels_test, scores)
best_f1 = -1.0
best_thresh = 0.5
for thr in thresholds:
    preds = (scores >= thr).astype(int)
    f1 = f1_score(labels_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thr

print(f"Best threshold by F1: {best_thresh:.4f}  F1: {best_f1:.4f}")

# Save threshold
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'w') as f:
    f.write(f"{best_thresh:.6f}\n")

print("Saved threshold to", OUT_PATH)
