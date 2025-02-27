from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import numpy as np

# Download YOLOv8 face detection model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# Load the model
model = YOLO(model_path)

# Define input image path
image_path = "/Users/bappa123/Desktop/Screenshot 2025-02-27 at 1.14.05 PM.png"

# Load image
image = Image.open(image_path)

# Run inference
output = model(image)

# Convert results to Detections format
results = Detections.from_ultralytics(output[0])

# Draw detections on the image (returns a NumPy array)
image_with_detections = output[0].plot()

# Convert NumPy array to PIL image
image_pil = Image.fromarray(image_with_detections.astype(np.uint8))

# Define output path
output_path = "/Users/bappa123/Desktop/output.png"

# Save the processed image
image_pil.save(output_path)

print(f"Saved output image at: {output_path}")
