import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import numpy as np
from loguru import logger
def face_detection_crop(input_dir):
    


    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

    model = YOLO(model_path)

      
    output_dir = ""
    output_txt_path = os.path.join(output_dir, "bounding_boxes.txt")
    os.makedirs(output_dir, exist_ok=True)  
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")): 
            image_path = os.path.join(input_dir, filename)
            
            image = Image.open(image_path)

            output = model(image)

            results = Detections.from_ultralytics(output[0])

            bounding_boxes = output[0].boxes.xywh.cpu().numpy()  

            output_image_path = os.path.join(output_dir, f"output_{filename}")
            left=0
            right=0
            bottom=0
            top=0

            
            with open(output_txt_path, "a") as f:
                for bbox in bounding_boxes:
                    x, y, w, h = bbox  
                    f.write(f"{image_path} -> {x} {y} {w} {h}\n")
                    left= int((x-w)/2)
                    right=int((x+w)/2)
                    bottom=int((y+h)/2)
                    top=int((y-h)/2)

            logger.info(f"Saved bounding box coordinates for {filename} at: {output_txt_path}")

            image_with_detections = output[0].plot()

            image_pil = Image.fromarray(image_with_detections.astype(np.uint8))
            face_crop = image_pil.crop((left, top, right, bottom))

            face_crop.save(output_image_path)

            logger.debug(f"Saved output image for {filename} at: {output_image_path}")
    logger.info("completed with face detection")
