import sys, os
# ensure the script's folder (src/) is on sys.path so local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import csv, os
import numpy as np
from model_improved import build_siamese
from preprocess import load_and_preprocess
from extract_features import extract_handwriting_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

PAIRS_CSV = r"E:\handwriting_matcher\pairs\train_pairs.csv"
MODEL_PATH = r"E:\handwriting_matcher\outputs\checkpoints\siamese_best.h5"
FUSION_MODEL = r"E:\handwriting_matcher\outputs\fusion_model.pkl"

print("Loading Siamese model...")
siamese = build_siamese()
siamese.load_weights(MODEL_PATH)

A, B, Y = [], [], []

print("Reading pairs...")
with open(PAIRS_CSV, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        A.append(row['imgA'])
        B.append(row['imgB'])
        Y.append(int(row['label']))

A = np.array(A)
B = np.array(B)
Y = np.array(Y)

features = []
labels = []

print("Extracting features...")
for i in range(len(A)):
    a_path, b_path = A[i], B[i]

    # CNN similarity score
    Aimg = np.expand_dims(load_and_preprocess(a_path),0)
    Bimg = np.expand_dims(load_and_preprocess(b_path),0)
    score = float(siamese.predict([Aimg,Bimg], verbose=0)[0][0])

    # handcrafted features
    featA = extract_handwriting_features(a_path)
    featB = extract_handwriting_features(b_path)

    diff = np.abs(np.array(featA) - np.array(featB))

    fused = [score] + diff.tolist()
    features.append(fused)
    labels.append(Y[i])

features = np.array(features)
labels = np.array(labels)

print("Training fusion classifier...")
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.15, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, max_depth=8)
clf.fit(X_train, y_train)

acc = clf.score(X_val, y_val)
print("Fusion accuracy:", acc)

joblib.dump(clf, FUSION_MODEL)
print("Fusion model saved to", FUSION_MODEL)
