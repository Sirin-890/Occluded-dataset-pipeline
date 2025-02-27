import cv2
from loguru import logger




def sunglass_occlusion(image_path, x, y, w, h, sunglass_path):
    """
    Overlays sunglasses onto the specified face region in a single image and returns the modified image.
    """
    image = cv2.imread(image_path)
    sunglass_image = cv2.imread(sunglass_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        logger.error(f"Error loading image: {image_path}")
        return None
    if sunglass_image is None:
        logger.error(f"Error loading sunglasses image: {sunglass_path}")
        return None
    
    # Resize sunglasses to match the face width
    sunglass_resized = cv2.resize(sunglass_image, (w, int(0.4 * h)))
    x_offset = x
    y_offset = y + int(0.3 * h)
    
    alpha_s = sunglass_resized[:, :, 3] / 255.0
    alpha_d = 1.0 - alpha_s
    for c in range(0, 3):
        image[y_offset:y_offset + sunglass_resized.shape[0], x_offset:x_offset + sunglass_resized.shape[1], c] = (
            alpha_s * sunglass_resized[:, :, c] + alpha_d * image[y_offset:y_offset + sunglass_resized.shape[0], x_offset:x_offset + sunglass_resized.shape[1], c]
        )
    
    return image
