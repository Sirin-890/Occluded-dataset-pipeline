import cv2
import random
import numpy as np
from loguru import logger

def percent_blackbox_single(image, x, y, w, h, occlusion_percent=10):
    """
    Applies a specified percentage of black patch occlusion to a face region in a single image.
    
    Args:
        image (numpy array): The input image.
        x (int): X-coordinate of the face bounding box.
        y (int): Y-coordinate of the face bounding box.
        w (int): Width of the face bounding box.
        h (int): Height of the face bounding box.
        occlusion_percent (float): Percentage of the face area to be occluded.
    
    Returns:
        numpy array: The modified image with occlusion.
    """
    if image is None:
        logger.error("Error loading image")
        return None
    
    face_area = w * h
    patch_area = int((occlusion_percent / 100) * face_area)  

    aspect_ratio = random.uniform(0.5, 2.0)
    patch_w = int((patch_area * aspect_ratio) ** 0.5)
    patch_h = int((patch_area / aspect_ratio) ** 0.5)

    patch_w = min(patch_w, w)
    patch_h = min(patch_h, h)

    patch_x = random.randint(x, x + w - patch_w)
    patch_y = random.randint(y, y + h - patch_h)

    image[patch_y:patch_y + patch_h, patch_x:patch_x + patch_w] = (0, 0, 0)
    
    return image
