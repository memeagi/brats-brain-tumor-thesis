import numpy as np
import cv2

def normalize(image):
    """
    Z-score normalization
    """
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / (std + 1e-8)

def resize(image, size=(240, 240)):
    """
    Resize image to fixed size
    """
    return cv2.resize(image, size)

def has_tumor(mask):
    """
    Check if slice contains tumor
    """
    return np.sum(mask) > 0