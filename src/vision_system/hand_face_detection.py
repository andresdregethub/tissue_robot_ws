
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================================
# 1. FILE PATHS
# ==========================================================

IMAGE_PATH = "images/hand_test.jpeg"

FACE_MODEL_PATH = "models/face_landmarker.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"

OUTPUT_PATH = "images/hand_face_annotated.jpg"


# ==========================================================
# 2. LOAD IMAGE
# ==========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

height, width, _ = image.shape

rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)


# ==========================================================
# 3. INITIALIZE FACE LANDMARKER
# ==========================================================

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=5
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)


# ==========================================================
# 4. INITIALIZE HAND LANDMARKER
# ==========================================================

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

hand_detector = vision.HandLandmarker.create_from_options(
    hand_options
)


# ==========================================================
# 5. DETECT FACE AND HANDS
# ==========================================================

face_results = face_detector.detect(mp_image)
hand_results = hand_detector.detect(mp_image)

face_count = len(face_results.face_landmarks)
hand_count = len(hand_results.hand_landmarks)

print(f"Faces detected: {face_count}")
print(f"Hands detected: {hand_count}")


# ==========================================================
# 6. FIND THE NOSE AND MOUTH REGION
# ==========================================================

face_detected = False
hand_covering_face = False

# These will store the nose-and-mouth region.
roi_x1 = roi_y1 = roi_x2 = roi_y2 = 0

if face_count > 0:

    face_detected = True

    # Use the first detected face.
    face_landmarks = face_results.face_landmarks[0]

    # Find the outer boundaries of the face.
    face_x = [
        int(point.x * width)
        for point in face_landmarks
    ]

    face_y = [
        int(point.y * height)
        for point in face_landmarks
    ]

    face_x1 = max(0, min(face_x))
    face_y1 = max(0, min(face_y))

    face_x2 = min(width - 1, max(face_x))
    face_y2 = min(height - 1, max(face_y))

    face_width = face_x2 - face_x1
    face_height = face_y2 - face_y1

    # Estimate the nose-and-mouth area.
    roi_x1 = int(face_x1 + face_width * 0.25)
    roi_x2 = int(face_x1 + face_width * 0.75)

    roi_y1 = int(face_y1 + face_height * 0.38)
    roi_y2 = int(face_y1 + face_height * 0.82)

    print("Face region located.")


# ==========================================================
# 7. CHECK WHETHER A HAND COVERS THE NOSE AND MOUTH
# ==========================================================

if face_detected and hand_count > 0:

    for hand_landmarks in hand_results.hand_landmarks:

        points_inside_region = 0

        hand_x = []
        hand_y = []

        # Examine all 21 hand landmarks.
        for point in hand_landmarks:

            x = int(point.x * width)
            y = int(point.y * height)

            hand_x.append(x)
            hand_y.append(y)

            # Check whether this landmark is inside
            # the estimated nose-and-mouth region.
            if (
                roi_x1 <= x <= roi_x2
                and roi_y1 <= y <= roi_y2
            ):
                points_inside_region += 1

        # A hand is considered near the face region
        # when at least 3 landmarks overlap it.
        if points_inside_region >= 3:

            hand_covering_face = True

        # Draw hand landmarks.
        for x, y in zip(hand_x, hand_y):

            cv2.circle(
                image,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        # Draw connections between hand landmarks.
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        for start, end in connections:

            cv2.line(
                image,
                (hand_x[start], hand_y[start]),
                (hand_x[end], hand_y[end]),
                (255, 0, 0),
                2
            )


# ==========================================================
# 8. DISPLAY DETECTION RESULTS
# ==========================================================

if face_detected:

    # Draw the estimated nose-and-mouth region.
    roi_color = (
        (0, 0, 255)
        if hand_covering_face
        else (0, 165, 255)
    )

    cv2.rectangle(
        image,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        roi_color,
        2
    )

    if hand_covering_face:

        status = "HAND NEAR NOSE/MOUTH"

    else:

        status = "NO HAND COVER DETECTED"

else:

    status = "NO FACE DETECTED"


cv2.putText(
    image,
    status,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255) if hand_covering_face else (0, 165, 255),
    2
)


# ==========================================================
# 9. SAVE RESULTS
# ==========================================================

cv2.imwrite(OUTPUT_PATH, image)

print(f"Detection result: {status}")
print(f"Annotated image saved to: {OUTPUT_PATH}")


# ==========================================================
# 10. CLEANUP
# ==========================================================

face_detector.close()
hand_detector.close()


