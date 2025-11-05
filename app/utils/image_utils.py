import cv2
import numpy as np 

def load_gray(image_path):
    """Load an image from a file path."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def denoise_and_threshold(image, blur_ksize=3):
    """Denoise, apply adaptive threshold — good for many scanned docs."""
    # Gaussian blur to smooth noise
    blurred_image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    # Adaptive thresholding keeps contrast in uneven lighting
    thresholded_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    return thresholded_image

def deskew_image(image):
    """Simple deskew using moments — helps when image is slightly rotated."""
    coords = np.column_stack(np.where(image < 255))
    if coords.size == 0:
        return image  # No content to deskew
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed_image = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
    return deskewed_image