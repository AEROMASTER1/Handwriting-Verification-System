import cv2
import numpy as np
from skimage.feature import hog
from scipy.stats import entropy

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img if img is not None else np.zeros((224,224), np.uint8)

def edge_density(img):
    edges = cv2.Canny(img, 50, 150)
    return np.mean(edges > 0)

def laplacian_variance(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return lap.var()

def pixel_intensity_mean(img):
    return float(np.mean(img))

def texture_energy(img):
    img_float = img.astype(np.float32) / 255.0
    gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1)
    return float(np.mean(gx**2 + gy**2))

def slant_angle(img):
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    if lines is None:
        return 0.0
    angles = [(theta - np.pi/2) for rho, theta in lines[:,0]]
    return float(np.mean(angles))

def hog_feature(img):
    img_resized = cv2.resize(img, (64,64))
    feat = hog(img_resized, orientations=8, pixels_per_cell=(16,16),
               cells_per_block=(1,1), visualize=False)
    return float(np.mean(feat))

def image_entropy(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).ravel()
    hist /= hist.sum() + 1e-9
    return float(entropy(hist))

def extract_handwriting_features(path):
    img = load_gray(path)
    return [
        edge_density(img),
        laplacian_variance(img),
        pixel_intensity_mean(img),
        texture_energy(img),
        slant_angle(img),
        hog_feature(img),
        image_entropy(img)
    ]
