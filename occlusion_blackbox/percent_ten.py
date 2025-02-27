import cv2
import os
import random
from loguru import logger

input_dir = "celebA_demo"
output_dir = "CV_blackbox_CelebA_10_percent"

os.makedirs(output_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            logger.error(f"Error loading image: {filename}")
            continue  

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_area = w * h
            patch_area = int(0.1 * face_area)  # 10% of the face area

            # Choose random patch dimensions (ensuring aspect ratio is reasonable)
            aspect_ratio = random.uniform(0.5, 2.0)  # Allow different shapes
            patch_w = int((patch_area * aspect_ratio) ** 0.5)
            patch_h = int((patch_area / aspect_ratio) ** 0.5)

            # Ensure patch fits within the face bounding box
            patch_w = min(patch_w, w)
            patch_h = min(patch_h, h)

            # Randomly position the patch inside the face bounding box
            patch_x = random.randint(x, x + w - patch_w)
            patch_y = random.randint(y, y + h - patch_h)

            # Draw the black patch
            cv2.rectangle(image, (patch_x, patch_y), (patch_x + patch_w, patch_y + patch_h), (0, 0, 0), -1)

            output_path = os.path.join(output_dir, f"{filename.split('.')[0]}_patched.jpg")
            cv2.imwrite(output_path, image)

print(f"Processing complete. Check the {output_dir} directory.")

def blackbox_10():
    pass
