# E:\handwriting_matcher\src\make_pairs.py
import os
import random
import csv
from glob import glob

# === Absolute paths for your system ===
DATA_DIR = r"E:\handwriting_matcher\data"                  # root folder with writer_xxx folders
OUT_CSV  = r"E:\handwriting_matcher\pairs\train_pairs.csv" # output CSV
# ======================================

# Parameters
NUM_POS_PER_WRITER = 30    # positive pairs per writer (adjustable)
NUM_NEG_PAIRS = 3000       # total negative pairs (adjustable)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

print("DATA_DIR =", DATA_DIR)
try:
    sample_listing = os.listdir(DATA_DIR)[:10]
except Exception as e:
    print("ERROR: cannot list DATA_DIR. Check the path. Exception:", e)
    raise SystemExit(1)
print("Sample folders inside DATA_DIR (first 10):", sample_listing)
print("Total items in DATA_DIR:", len(os.listdir(DATA_DIR)))

def get_writers(data_dir):
    return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

def build_pairs(data_dir, out_csv, num_pos_per_writer=30, num_neg_pairs=3000):
    writers = get_writers(data_dir)
    print(f"Found {len(writers)} writer folders.")
    writer_imgs = {}
    all_imgs = []

    exts = ['*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp']
    for w in writers:
        imgs = []
        for e in exts:
            imgs.extend(glob(os.path.join(data_dir, w, e)))
        imgs = sorted(imgs)
        # debug print for first few writers
        # if w.startswith('writer_0001') or w.endswith('0001'):
        #     print(w, "->", imgs[:5])
        if len(imgs) >= 1:
            writer_imgs[w] = imgs
            all_imgs.extend(imgs)

    # how many writers have >=2 images?
    writers_with_2 = [w for w, imgs in writer_imgs.items() if len(imgs) >= 2]
    print(f"Writers with >=2 images: {len(writers_with_2)}")

    if len(writers_with_2) == 0:
        print("Not enough writers with >=2 images. Exiting.")
        return

    pairs = []

    # Positive pairs
    for w, imgs in writer_imgs.items():
        if len(imgs) < 2:
            continue
        combos = []
        n = len(imgs)
        for i in range(n):
            for j in range(i+1, n):
                combos.append((imgs[i], imgs[j]))
        random.shuffle(combos)
        take = min(num_pos_per_writer, len(combos))
        for a, b in combos[:take]:
            pairs.append([a, b, 1])

    # Negative pairs
    writers_list = list(writer_imgs.keys())
    neg_count = 0
    attempts = 0
    max_attempts = num_neg_pairs * 10
    while neg_count < num_neg_pairs and attempts < max_attempts:
        w1, w2 = random.sample(writers_list, 2)
        a = random.choice(writer_imgs[w1])
        b = random.choice(writer_imgs[w2])
        pairs.append([a, b, 0])
        neg_count += 1
        attempts += 1

    random.shuffle(pairs)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['imgA', 'imgB', 'label'])
        writer.writerows(pairs)

    print(f"Wrote {len(pairs)} pairs to {out_csv}")
    print("Sample rows (first 5):")
    for row in pairs[:5]:
        print(row)

if __name__ == "__main__":
    build_pairs(DATA_DIR, OUT_CSV, NUM_POS_PER_WRITER, NUM_NEG_PAIRS)
