# src/preview_pairs.py
import csv, os
import numpy as np
from PIL import Image
from preprocess import load_and_preprocess
import matplotlib.pyplot as plt

PAIRS_CSV = r"E:\handwriting_matcher\pairs\train_pairs.csv"
OUT_DIR = r"E:\handwriting_matcher\outputs\previews"
os.makedirs(OUT_DIR, exist_ok=True)

def read_pairs(csv_path, max_rows=8):
    pairs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= max_rows: break
            pairs.append((r['imgA'], r['imgB'], int(r['label'])))
    return pairs

pairs = read_pairs(PAIRS_CSV, max_rows=8)
print("Loaded", len(pairs), "pairs for preview\n")

for idx, (a,b,label) in enumerate(pairs):
    A = load_and_preprocess(a)
    B = load_and_preprocess(b)
    print(f"Pair {idx+1}:")
    print("  A:", a, "->", A.shape, "min/max:", float(A.min()), float(A.max()))
    print("  B:", b, "->", B.shape, "min/max:", float(B.min()), float(B.max()))
    print("  label:", label)
    # save a small visual concatenation
    concat = np.concatenate([A.squeeze(), B.squeeze()], axis=1)  # side by side
    # to PIL image
    img = Image.fromarray((concat * 255).astype('uint8'))
    outpath = os.path.join(OUT_DIR, f"preview_{idx+1}_label{label}.png")
    img.save(outpath)
    print("  preview saved ->", outpath)
    print()
