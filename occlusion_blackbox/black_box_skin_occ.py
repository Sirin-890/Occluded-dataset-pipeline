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
    # cv2.imshow('Occluded Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
def skin_colour(img_path, occlusion_percent):
   
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_skin = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

   
    skin_mask = cv2.inRange(img_skin, lower, upper)

    
    skin_pix_val = img[skin_mask > 0]

   
    if skin_pix_val.size == 0:
        logger.error("skin nahi mili")
    else:
    
        r_pix, g_pix, b_pix = np.mean(skin_pix_val, axis=0)

    
    h, w, _ = img.shape
    
    
    total_area = h * w
    occlusion_area = int(total_area * (occlusion_percent / 100))
    
    occluded_area = 0
    
    while occluded_area < occlusion_area:
        
        block_w = random.randint(10, min(100, w // 4))
        block_h = random.randint(10, min(100, h // 4))
        
        
        if occluded_area + (block_w * block_h) > occlusion_area:
            break
        
       
        x = random.randint(0, w - block_w)
        y = random.randint(0, h - block_h)
        
        
        img[y:y + block_h, x:x + block_w] = (r_pix,g_pix,b_pix)
        occluded_area += (block_w * block_h)

    occluded_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, occluded_bgr)

    
    # cv2.imshow('Occluded Image', occluded_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    logger.info("slkin occlusion done done")
# if __name__=="__main__":
#     pth=""
#     skin_colour(pth,80)

# import cv2
# import random
# import numpy as np
# from loguru import logger
# from PIL import Image
# import csv

output_csv = 'result.csv'
output2_csv = 'result2.csv'

def percent_blackbox_region(img_path, occlusion_percent, region='random'):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    h, w, _ = img.shape

    total_area = h * w
    occlusion_area = int(total_area * (occlusion_percent / 100))
    occluded_area = 0

    region_choice = region
    if region == 'random':
        region_choice = random.choice(['upper', 'lower'])

    while occluded_area < occlusion_area:
        block_w = random.randint(10, min(100, w // 4))
        block_h = random.randint(10, min(100, h // 4))
        if occluded_area + (block_w * block_h) > occlusion_area:
            break
        x = random.randint(0, w - block_w)
        if region_choice == 'upper':
            y = random.randint(0, h // 2 - block_h)
        else:  # lower
            y = random.randint(h // 2, h - block_h)

        img[y:y + block_h, x:x + block_w] = (0, 0, 0)
        occluded_area += (block_w * block_h)

    cv2.imwrite(img_path, img)

def skin_colour_region(img_path, occlusion_percent, region='random'):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_skin = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    skin_mask = cv2.inRange(img_skin, lower, upper)
    skin_pix_val = img[skin_mask > 0]

    if skin_pix_val.size == 0:
        logger.error("No skin detected")
        return

    r_pix, g_pix, b_pix = np.mean(skin_pix_val, axis=0)
    h, w, _ = img.shape
    total_area = h * w
    occlusion_area = int(total_area * (occlusion_percent / 100))
    occluded_area = 0

    region_choice = region
    if region == 'random':
        region_choice = random.choice(['upper', 'lower'])

    while occluded_area < occlusion_area:
        block_w = random.randint(10, min(100, w // 4))
        block_h = random.randint(10, min(100, h // 4))
        if occluded_area + (block_w * block_h) > occlusion_area:
            break
        x = random.randint(0, w - block_w)
        if region_choice == 'upper':
            y = random.randint(0, h // 2 - block_h)
        else:
            y = random.randint(h // 2, h - block_h)

        img[y:y + block_h, x:x + block_w] = (r_pix, g_pix, b_pix)
        occluded_area += (block_w * block_h)

    occluded_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, occluded_bgr)
    logger.info("Skin occlusion done")

if __name__ == "__main__":
    pth = "your_image_path.jpg"
    skin_colour(pth, 80, region='upper')  # Choose 'upper', 'lower', or 'random'



