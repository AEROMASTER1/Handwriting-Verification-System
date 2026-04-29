# src/train.py (Using Improved MobileNetV2 Siamese Model)

import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from preprocess import load_and_preprocess
from tensorflow.keras.utils import Sequence

# 👉 IMPORTANT: use the improved model
from model_improved import build_siamese

PAIRS_CSV = r"E:\handwriting_matcher\pairs\train_pairs.csv"
MODEL_PATH = r"E:\handwriting_matcher\outputs\checkpoints\siamese_best.h5"

IMG_SHAPE = (105, 105, 1)
BATCH_SIZE = 16
EPOCHS = 10
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# --- Load CSV ---
def load_pairs(csv_path):
    A, B, Y = [], [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            A.append(row["imgA"])
            B.append(row["imgB"])
            Y.append(int(row["label"]))
    return np.array(A), np.array(B), np.array(Y)


# --- Keras Sequence (batch loader) ---
class PairSequence(Sequence):
    def __init__(self, A_paths, B_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.A_paths = A_paths
        self.B_paths = B_paths
        self.labels = labels
        self.batch_size = batch_size
        self.indices = np.arange(len(A_paths))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return max(1, int(np.ceil(len(self.A_paths) / float(self.batch_size))))

    def __getitem__(self, idx):
        batch_idxs = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        A_batch = np.stack([load_and_preprocess(p) for p in self.A_paths[batch_idxs]], axis=0)
        B_batch = np.stack([load_and_preprocess(p) for p in self.B_paths[batch_idxs]], axis=0)
        Y_batch = self.labels[batch_idxs]

        # MUST RETURN TUPLE for TensorFlow
        return (A_batch, B_batch), Y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# --- TRAINING ---
if __name__ == "__main__":
    print("Loading pairs...")
    A, B, Y = load_pairs(PAIRS_CSV)

    print("Total pairs:", len(A))

    A_train, A_val, B_train, B_val, Y_train, Y_val = train_test_split(
        A, B, Y, test_size=0.15, random_state=RANDOM_SEED
    )

    print("Training pairs:", len(A_train))
    print("Validation pairs:", len(A_val))

    train_seq = PairSequence(A_train, B_train, Y_train, batch_size=BATCH_SIZE)
    val_seq = PairSequence(A_val, B_val, Y_val, batch_size=BATCH_SIZE, shuffle=False)

    print("\nBuilding improved MobileNetV2 Siamese model...\n")
    model = build_siamese()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss")
    earlystop = EarlyStopping(patience=3, restore_best_weights=True)

    print("\nStarting training...\n")

    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        callbacks=[checkpoint, earlystop],
        verbose=1
    )

    print("\nTraining finished.")
    print("Model saved at:", MODEL_PATH)
