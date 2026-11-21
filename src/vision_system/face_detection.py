

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================================
# 1. FILE PATHS
# ==========================================================

IMAGE_PATH = "images/face_1.jpg"
MODEL_PATH = "models/face_landmarker.task"
OUTPUT_PATH = "images/annotated.jpg"


# ==========================================================
# 2. LOAD IMAGE
# ==========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)


# ==========================================================
# 3. INITIALIZE FACE LANDMARKER
# ==========================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=5
)

detector = vision.FaceLandmarker.create_from_options(options)


# ==========================================================
# 4. DETECT FACES
# ==========================================================

results = detector.detect(mp_image)

face_count = len(results.face_landmarks)

print(f"Faces detected: {face_count}")


# ==========================================================
# 5. DRAW FACIAL LANDMARKS
# ==========================================================

height, width, _ = image.shape

for face_landmarks in results.face_landmarks:

    x_coordinates = []
    y_coordinates = []

    for landmark in face_landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        x_coordinates.append(x)
        y_coordinates.append(y)

        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    x_min = max(0, min(x_coordinates))
    y_min = max(0, min(y_coordinates))
    x_max = min(width - 1, max(x_coordinates))
    y_max = min(height - 1, max(y_coordinates))

    cv2.rectangle(
        image,
        (x_min, y_min),
        (x_max, y_max),
        (255, 0, 0),
        2
    )


# ==========================================================
# 6. SAVE RESULT
# ==========================================================

cv2.imwrite(OUTPUT_PATH, image)

print(f"Annotated image saved to: {OUTPUT_PATH}")

detector.close()
