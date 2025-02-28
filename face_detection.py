import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
import numpy as np
from loguru import logger

def face_detection_crop(image):
    """
    Detects a face in a single image, crops it, and returns the cropped face image.
    """
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    model = YOLO(model_path)
    
    #image = Image.open(image_path)
    output = model(image)
    results = Detections.from_ultralytics(output[0])
    bounding_boxes = output[0].boxes.xywh.cpu().numpy()
    
    if len(bounding_boxes) == 0:
        logger.warning("No face detected in the image.")
        return None
    
    x, y, w, h = bounding_boxes[0]  
    left = int(x - w / 2)
    right = int(x + w / 2)
    top = int(y - h / 2)
    bottom = int(y + h / 2)
    
    face_crop = image.crop((left, top, right, bottom))
    
    logger.info("Face detection and cropping completed.")
    return face_crop,x,y,w,h
