import cv2
import cvzone
import os
import dlib
from loguru import logger

def apply_sunglasses(face_image_path, overlay_path):
    face_image=cv2.imread(face_image_path)
    
    
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise FileNotFoundError(f"Error: '{overlay_path}' not found.")
    
    
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
   
    predictor_path = "occlusion_blackbox/shape_predictor_68_face_landmarks.dat"  # Ensure this file exists
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    
    faces = detector(gray)
    if len(faces) == 0:
        logger.error("No faces detected!")
        return face_image
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        
        left_eye_x = (landmarks.part(36).x + landmarks.part(39).x) // 2
        left_eye_y = (landmarks.part(36).y + landmarks.part(39).y) // 2
        right_eye_x = (landmarks.part(42).x + landmarks.part(45).x) // 2
        right_eye_y = (landmarks.part(42).y + landmarks.part(45).y) // 2
        
        
        width = int(2 * (right_eye_x - left_eye_x))  
        height = int(width * 1.2) 
        
        
        overlay_resized = cv2.resize(overlay, (width, height))
        
        
        x = left_eye_x - int(width * 0.3)  
        y = left_eye_y - int(height * 0.5)
        
       
        face_image = cvzone.overlayPNG(face_image, overlay_resized, [x, y])
        cv2.imwrite(face_image_path,face_image)
    
    

# Example usage
if __name__ == "__main__":
    input_folder = "cleeba"  # Folder containing cropped face images
    output_folder = "output_glasses"  # Folder to save processed images
    os.makedirs(output_folder, exist_ok=True)
    
    overlay_path = "glass8.png"  # Path to sunglasses image
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, filename)
            face_img = cv2.imread(image_path)
            if face_img is None:
                print(f"Error: Could not load {filename}. Skipping...")
                continue
            
            processed_img = apply_sunglasses(face_img, overlay_path)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_img)
            print(f"Processed and saved: {output_path}")
    
    print("Processing complete!")