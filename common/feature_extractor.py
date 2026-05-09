# -*- coding: utf-8 -*-
# common/feature_extractor.py
# Trích xuất đặc trưng từ ảnh: histogram màu, HOG, và flatten để dùng cho sklearn

import cv2
import numpy as np
from skimage.feature import hog

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Trích xuất histogram màu 3 kênh (HSV hoặc RGB)
    Args:
        image: ảnh dạng numpy array (BGR hoặc RGB)
        bins: số bin cho mỗi kênh
    Returns:
        vector đặc trưng 1D
    """
    # Chuyển sang HSV (thường tốt hơn cho màu sắc)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_hog(image):
    """
    Trích xuất đặc trưng HOG (Histogram of Oriented Gradients)
    Args:
        image: ảnh grayscale
    Returns:
        vector HOG
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize về kích thước cố định để HOG có cùng số chiều
    resized = cv2.resize(gray, (128, 128))
    hog_features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys')
    return hog_features

def extract_features(image_path):
    """
    Kết hợp cả histogram và HOG
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    hist = extract_color_histogram(img)
    hog = extract_hog(img)
    features = np.concatenate([hist, hog])
    return features