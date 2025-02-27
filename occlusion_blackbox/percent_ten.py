import cv2
import random
import numpy as np
from loguru import logger

def percent_ten_blackbox_single(image_path, x, y, w, h):
    """
    Applies a 10% black patch to a specified face region in a single image and returns the modified image.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Error loading image: {image_path}")
        return None
    
    face_area = w * h
    patch_area = int(0.1 * face_area)  # 10% of the face area

    # Choose random patch dimensions (ensuring aspect ratio is reasonable)
    aspect_ratio = random.uniform(0.5, 2.0)
    patch_w = int((patch_area * aspect_ratio) ** 0.5)
    patch_h = int((patch_area / aspect_ratio) ** 0.5)

    # Ensure patch fits within the face bounding box
    patch_w = min(patch_w, w)
    patch_h = min(patch_h, h)

    # Randomly position the patch inside the face bounding box
    patch_x = random.randint(x, x + w - patch_w)
    patch_y = random.randint(y, y + h - patch_h)

    # Draw the black patch
    image[patch_y:patch_y + patch_h, patch_x:patch_x + patch_w] = (0, 0, 0)
    
    return image