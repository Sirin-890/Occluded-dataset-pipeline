import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import numpy as np
from loguru import logger
def face_detection():

    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

    model = YOLO(model_path)

    input_dir = ""  
    output_dir = ""
    os.makedirs(output_dir, exist_ok=True)  
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")): 
            image_path = os.path.join(input_dir, filename)
            
            image = Image.open(image_path)

            output = model(image)

            results = Detections.from_ultralytics(output[0])

            bounding_boxes = output[0].boxes.xywh.cpu().numpy()  

            output_image_path = os.path.join(output_dir, f"output_{filename}")
            output_txt_path = os.path.join(output_dir, f"bounding_boxes_{os.path.splitext(filename)[0]}.txt")

            with open(output_txt_path, "w") as f:
                for bbox in bounding_boxes:
                    x, y, w, h = bbox  
                    f.write(f"{image_path} -> {x} {y} {w} {h}\n")

            print(f"Saved bounding box coordinates for {filename} at: {output_txt_path}")

            image_with_detections = output[0].plot()

            image_pil = Image.fromarray(image_with_detections.astype(np.uint8))

            image_pil.save(output_image_path)

            print(f"Saved output image for {filename} at: {output_image_path}")
    logger.info("completed with face detection")
