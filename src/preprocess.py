import cv2
import numpy as np
def get_grayscale(image):
    """Converts image to grayscale for simpler processing."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def reduce_noise(image):
    """
    Applies Bilateral Filtering. 
    Unlike Gaussian blur, it removes noise while preserving the 
    sharp edges of the alphanumeric characters.
    """
    return cv2.bilateralFilter(image, 11, 17, 17)

def enhance_contrast(image):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    This is highly effective for Indian plates under bright sunlight 
    or low-light glare.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def thresholding(image):
    """
    Converts to a binary (black and white) image. 
    Using Adaptive Thresholding helps when one part of the plate is 
    shadowed and the other is bright.
    """
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

def prepare_for_ocr(plate_img):
    """
    The full pipeline as described in your 'Methodology' section.
    1. Grayscale -> 2. Noise Reduction -> 3. Contrast -> 4. Threshold
    """
    gray = get_grayscale(plate_img)
    smooth = reduce_noise(gray)
    contrast = enhance_contrast(smooth)
    # You can return 'contrast' or 'thresh' depending on which 
    # performs better with your specific OCR (EasyOCR/PaddleOCR)
    thresh = thresholding(contrast)
    
    return thresh
