# Let's start by installing the required library
# pip install opencv-python

import cv2


def main():
    # Load the sunglasses image with alpha channel (transparency)
    sunglass_image = cv2.imread(
        "D:\Face_occluded Pipeline\Occluded-dataset-pipeline\sunglasses.png",
        cv2.IMREAD_UNCHANGED,
    )
    destination_image = cv2.imread(
        "D:\Face_occluded Pipeline\Occluded-dataset-pipeline\img3.png"
    )

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray_image = cv2.cvtColor(destination_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No face detected in the destination image.")
        return

    x, y, w, h = faces[0]

    # Resize the sunglass image to match the face's width and adjust the position
    sunglass_resized = cv2.resize(sunglass_image, (w, int(0.4 * h)))
    x_offset = x
    y_offset = y + int(0.3 * h)

    alpha_s = sunglass_resized[:, :, 3] / 255.0
    alpha_d = 1.0 - alpha_s
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

    cv2.imwrite("output_image3.jpg", destination_image)

    print("Sunglass attached!")


if __name__ == "__main__":
    main()
