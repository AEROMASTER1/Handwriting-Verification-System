# src/preprocess.py
from PIL import Image
import numpy as np

IMG_SIZE = (105,105)
  # size used for siamese models

def load_and_preprocess(path, img_size=IMG_SIZE):
    """
    Loads an image (handles png/jpg/tif) -> converts to grayscale -> resize -> normalize -> (h,w,1)
    Returns numpy array dtype float32 in range [0,1].
    """
    img = Image.open(path).convert('L')          # convert to grayscale
    img = img.resize(img_size)                   # resize
    arr = np.asarray(img, dtype='float32') / 255.0
    # if image is shape (h,w), add channel axis
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr
