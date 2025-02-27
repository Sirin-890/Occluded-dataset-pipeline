import cv2
import os

# Load the sunglasses image with alpha channel
sunglass_path = r"sunglasses.png"
sunglass_image = cv2.imread(sunglass_path, cv2.IMREAD_UNCHANGED)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def apply_sunglasses(image_path, output_path):
    destination_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(destination_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return

    x, y, w, h = faces[0]

    # Resize sunglass image to match face width
    sunglass_resized = cv2.resize(sunglass_image, (w, int(0.4 * h)))
    x_offset = x
    y_offset = y + int(0.3 * h)

    alpha_s = sunglass_resized[:, :, 3] / 255.0
    alpha_d = 1.0 - alpha_s

    # Overlay sunglasses on the face
    for c in range(0, 3):
        destination_image[
            y_offset : y_offset + sunglass_resized.shape[0],
            x_offset : x_offset + sunglass_resized.shape[1],
            c,
        ] = (
            alpha_s * sunglass_resized[:, :, c]
            + alpha_d
            * destination_image[
                y_offset : y_offset + sunglass_resized.shape[0],
                x_offset : x_offset + sunglass_resized.shape[1],
                c,
            ]
        )

    cv2.imwrite(output_path, destination_image)
    print(f"Sunglasses added: {output_path}")


def process_images(input_dir):
    output_dir = os.path.join(input_dir, "output_images")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")
            apply_sunglasses(input_path, output_path)

    print(f"All images processed. Output directory: {output_dir}")
    return output_dir


if __name__ == "__main__":
    input_dir = r"images"
    output_folder = process_images(input_dir)
    print("Output Directory:", output_folder)
