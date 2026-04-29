# src/evaluate.py (robust: rebuild model then load weights)
import csv, os, numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess

# Use the model builder (must exist)
from model_improved import build_siamese

PAIRS_CSV = r"E:\handwriting_matcher\pairs\train_pairs.csv"
MODEL_PATH = r"E:\handwriting_matcher\outputs\checkpoints\siamese_best.h5"
OUT_DIR = r"E:\handwriting_matcher\outputs\eval"
os.makedirs(OUT_DIR, exist_ok=True)

# Load all pairs (we'll use the validation split used in train.py)
pairs = []
labels = []
with open(PAIRS_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        pairs.append((r['imgA'], r['imgB']))
        labels.append(int(r['label']))
pairs = np.array(pairs)
labels = np.array(labels)

# Use last ~15% as test (same as training split)
n = len(labels)
start = int(n * 0.85)
pairs_test = pairs[start:]
labels_test = labels[start:]
print("Total pairs:", n, "Using test pairs:", len(labels_test))

# --- Rebuild model and load weights (preferred) ---
print("Rebuilding model from code...")
model = build_siamese()
# Try loading weights from the HDF5 checkpoint (works if file contains weights)
try:
    model.load_weights(MODEL_PATH)
    print("Weights loaded into rebuilt model from:", MODEL_PATH)
except Exception as e:
    print("Could not load weights into rebuilt model:", e)
    print("Attempting to load full model as fallback (unsafe deserialization may be required)...")
    # Fallback: try load_model with safe_mode disabled (only if you trust the file)
    try:
        # This may raise if environment disallows unsafe deserialization
        tf.keras.utils.get_custom_objects()  # ensure utils loaded
        model = load_model(MODEL_PATH, compile=False)
        print("Loaded full model via load_model fallback.")
    except Exception as e2:
        print("Fallback load_model also failed:", e2)
        raise SystemExit("Unable to load model. See error above.")

# Predict scores
scores = []
for a,b in pairs_test:
    A = np.expand_dims(load_and_preprocess(a),0)
    B = np.expand_dims(load_and_preprocess(b),0)
    s = float(model.predict([A,B], verbose=0)[0][0])
    scores.append(s)
scores = np.array(scores)
preds = (scores >= 0.5).astype(int)

# Metrics
acc = accuracy_score(labels_test, preds)
prec = precision_score(labels_test, preds, zero_division=0)
rec = recall_score(labels_test, preds, zero_division=0)
f1 = f1_score(labels_test, preds, zero_division=0)
cm = confusion_matrix(labels_test, preds)
try:
    auc = roc_auc_score(labels_test, scores)
except:
    auc = float('nan')

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC: {auc:.4f}")
print("Confusion matrix:\n", cm)

# Save confusion matrix figure
plt.figure(figsize=(4,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
ticks = [0,1]
plt.xticks(ticks, ['Not match','Match'])
plt.yticks(ticks, ['Not match','Match'])
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close()

# ROC curve
if not np.isnan(auc):
    fpr, tpr, _ = roc_curve(labels_test, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
    plt.close()

# Save a small CSV with per-pair scores (first 200)
out_csv = os.path.join(OUT_DIR, "scores_sample.csv")
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    import csv
    w = csv.writer(f)
    w.writerow(['imgA','imgB','label','score','pred'])
    for i in range(min(200, len(labels_test))):
        a,b = pairs_test[i]
        w.writerow([a,b,int(labels_test[i]), float(scores[i]), int(preds[i])])

print("Saved confusion matrix and ROC (if computed) to", OUT_DIR)
