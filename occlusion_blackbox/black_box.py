import cv2
import random
import numpy as np
from loguru import logger
from PIL import Image
import csv
output_csv='result.csv'
output2_csv='result2.csv'

def percent_ten_blackbox_single(img_path, occlusion_percent):
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")

    h, w, _ = img.shape
    
    # Calculate occlusion area size based on percentage
    total_area = h * w
    occlusion_area = int(total_area * (occlusion_percent / 100))
    
    occluded_area = 0
    
    while occluded_area < occlusion_area:
        # Randomly choose the size of the occlusion block
        block_w = random.randint(10, min(100, w // 4))
        block_h = random.randint(10, min(100, h // 4))
        
        # Ensure we don't exceed the target occlusion area
        if occluded_area + (block_w * block_h) > occlusion_area:
            break
        
        # Random position for the block
        x = random.randint(0, w - block_w)
        y = random.randint(0, h - block_h)
        
        # Fill the block with black
        img[y:y + block_h, x:x + block_w] = (0, 0, 0)
        occluded_area += (block_w * block_h)

    # Save the modified image back to the original path
    cv2.imwrite(img_path, img)

    # Optionally show the result
    cv2.imshow('Occluded Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()