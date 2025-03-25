import cv2
from loguru import logger




def sunglass_occlusion(img_path, sunglasses_path):
    # Load the face image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    
    # Load the sunglasses image with alpha channel (for transparency)
    sunglasses = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)
    if sunglasses is None:
        raise FileNotFoundError(f"Sunglasses image not found at path: {sunglasses_path}")

    h, w, _ = img.shape
    
    # Scale sunglasses to about 70–80% of the face width (more natural)
    glasses_width = int(w * 0.75)
    glasses_height = int(glasses_width * (sunglasses.shape[0] / sunglasses.shape[1]))
    
    resized_sunglasses = cv2.resize(sunglasses, (glasses_width, glasses_height))

    # Adjust position closer to the center of the face
    x_offset = (w - glasses_width) // 2
    y_offset = int(h * 0.45)  # Placing it around 45% down the face

    for i in range(glasses_height):
        for j in range(glasses_width):
            if y_offset + i < h and x_offset + j < w:
                # Extract alpha channel for transparency
                alpha = resized_sunglasses[i, j, 3] / 255.0
                if alpha > 0:
                    img[y_offset + i, x_offset + j] = (
                        alpha * resized_sunglasses[i, j, :3] +
                        (1 - alpha) * img[y_offset + i, x_offset + j]
                    )

    # Save the modified image back to the original path
    cv2.imwrite(img_path, img)

    # Show the result
    cv2.imshow('Image with Sunglasses', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()