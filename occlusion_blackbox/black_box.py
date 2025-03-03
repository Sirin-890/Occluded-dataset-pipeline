import cv2
import random
import numpy as np
from loguru import logger
from PIL import Image
import csv
output_csv='result.csv'
output2_csv='result2.csv'

def percent_ten_blackbox_single(image, x, y, w, h, occlusion_percent=10):
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
    image_np = np.array(image)

    
    if len(image_np.shape) == 2:  
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    face_area = w * h
    patch_area = int((occlusion_percent / 100) * face_area)  

    aspect_ratio = random.uniform(0.5, 2.0)
    patch_w = int((patch_area * aspect_ratio) ** 0.5)
    patch_h = int((patch_area / aspect_ratio) ** 0.5)

    patch_w = int(min(patch_w, w))
    patch_h = int(min(patch_h, h))

    patch_x = random.randint(int(x), int(x + w - patch_w))
    patch_y = random.randint(int(y), int(y + h - patch_h))
    logger.error(patch_h)
    logger.error(patch_y)
    logger.error(patch_x)
    logger.error(patch_w)
    logger.error(image_np)
    image_np_flat=image_np.reshape(-1,image_np.shape[-1])
    np.savetxt(output_csv,image_np_flat,delimiter=",",fmt="%d")
    cv2.rectangle(image_np, (patch_x, patch_y), (patch_x + patch_w, patch_y + patch_h), (0, 0, 0), -1)

    #image_np[patch_y:patch_y + patch_h, patch_x:patch_x + patch_w,:] = 0
    #image2_np_flat=image_np.reshape(-1,image_np.shape[-1])

    #np.savetxt(output2_csv,image2_np_flat,delimiter=",",fmt="%d")
    #logger.error(image_np)

    image_pil = Image.fromarray(image_np)

    logger.debug("Photo ready with occlusion")

    return image_pil