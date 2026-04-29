import os, numpy as np, csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from preprocess import load_and_preprocess
from model_improved import build_siamese
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

PAIRS_CSV = r"E:\handwriting_matcher\pairs\train_pairs.csv"
OLD_MODEL = r"E:\handwriting_matcher\outputs\checkpoints\siamese_best.h5"
NEW_MODEL = r"E:\handwriting_matcher\outputs\checkpoints\siamese_finetuned.h5"

BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
RANDOM_SEED = 42

# --- PairSequence (same from your train.py)
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
        return int(np.ceil(len(self.A_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        A = np.stack([load_and_preprocess(p) for p in self.A_paths[batch_idx]])
        B = np.stack([load_and_preprocess(p) for p in self.B_paths[batch_idx]])
        Y = self.labels[batch_idx]

        return (A, B), Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# --- Load pairs
A, B, Y = [], [], []
with open(PAIRS_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        A.append(row["imgA"])
        B.append(row["imgB"])
        Y.append(int(row["label"]))
A = np.array(A)
B = np.array(B)
Y = np.array(Y)

# --- Train & val split
A_train, A_val, B_train, B_val, Y_train, Y_val = train_test_split(
    A, B, Y, test_size=0.15, random_state=RANDOM_SEED
)

train_seq = PairSequence(A_train, B_train, Y_train)
val_seq = PairSequence(A_val, B_val, Y_val, shuffle=False)

# --- Load model and weights
model = build_siamese()
model.load_weights(OLD_MODEL)

# --- Fine-tuning: unfreeze entire model
model.trainable = True

# compile with low learning rate
model.compile(optimizer=Adam(LR), loss="binary_crossentropy", metrics=["accuracy"])

# --- Callbacks
os.makedirs(os.path.dirname(NEW_MODEL), exist_ok=True)
ckpt = ModelCheckpoint(NEW_MODEL, save_best_only=True, monitor="val_loss")
early = EarlyStopping(patience=2, restore_best_weights=True)

# --- TRAINING
print("\nStarting fine-tuning...\n")
model.fit(train_seq, validation_data=val_seq, epochs=EPOCHS, callbacks=[ckpt, early], verbose=1)

print("\nFine-tuned model saved to:", NEW_MODEL)
