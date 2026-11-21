import cv2
import mediapipe as mp
import numpy as np
import socket
import struct
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================================
# 1. NETWORK SETTINGS
# ==========================================================

HOST = "0.0.0.0"
PORT = 9999

FACE_MODEL_PATH = "models/face_landmarker.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"


# ==========================================================
# 2. RECEIVE COMPLETE DATA
# ==========================================================

def receive_exact(connection, size):

    data = b""

    while len(data) < size:

        packet = connection.recv(size - len(data))

        if not packet:
            return None

        data += packet

    return data


# ==========================================================
# 3. INITIALIZE MEDIAPIPE
# ==========================================================

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=5
)

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)

hand_detector = vision.HandLandmarker.create_from_options(
    hand_options
)


# ==========================================================
# 4. START TCP SERVER
# ==========================================================

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server.bind((HOST, PORT))
server.listen(1)

print(f"Waiting for Windows camera on port {PORT}...")
print("Start the Windows sender to connect.")


# ==========================================================
# 5. ACCEPT WINDOWS CONNECTION
# ==========================================================

connection, address = server.accept()

print(f"Connected to: {address}")


# ==========================================================
# 6. PROCESS LIVE VIDEO
# ==========================================================

previous_timestamp = 0

try:

    while True:

        # Receive the size of the next JPEG frame.
        size_data = receive_exact(connection, 4)

        if size_data is None:
            break

        frame_size = struct.unpack("!I", size_data)[0]

        if frame_size <= 0 or frame_size > 10_000_000:
            raise ValueError("Invalid video frame size")

        # Receive the actual JPEG frame.
        frame_data = receive_exact(connection, frame_size)

        if frame_data is None:
            break

        # Decode JPEG into an OpenCV image.
        encoded = np.frombuffer(frame_data, dtype=np.uint8)

        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        if image is None:
            continue

        height, width, _ = image.shape

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        # MediaPipe VIDEO mode requires increasing timestamps.
        timestamp = max(
            int(time.monotonic() * 1000),
            previous_timestamp + 1
        )

        previous_timestamp = timestamp

        # Run face and hand detection.
        faces = face_detector.detect_for_video(
            mp_image, timestamp
        )

        hands = hand_detector.detect_for_video(
            mp_image, timestamp
        )

        face_count = len(faces.face_landmarks)
        hand_count = len(hands.hand_landmarks)

        hand_near_face = False

        # --------------------------------------------------
        # FIND FACE AND NOSE/MOUTH REGION
        # --------------------------------------------------

        if face_count > 0:

            landmarks = faces.face_landmarks[0]

            face_x = [int(p.x * width) for p in landmarks]
            face_y = [int(p.y * height) for p in landmarks]

            x1 = max(0, min(face_x))
            y1 = max(0, min(face_y))
            x2 = min(width - 1, max(face_x))
            y2 = min(height - 1, max(face_y))

            face_width = x2 - x1
            face_height = y2 - y1

            roi_x1 = int(x1 + face_width * 0.25)
            roi_x2 = int(x1 + face_width * 0.75)
            roi_y1 = int(y1 + face_height * 0.38)
            roi_y2 = int(y1 + face_height * 0.82)

            # --------------------------------------------------
            # CHECK HAND PROXIMITY
            # --------------------------------------------------

            for hand in hands.hand_landmarks:

                points_inside = 0

                hand_points = []

                for point in hand:

                    x = int(point.x * width)
                    y = int(point.y * height)

                    hand_points.append((x, y))

                    if (
                        roi_x1 <= x <= roi_x2
                        and roi_y1 <= y <= roi_y2
                    ):
                        points_inside += 1

                if points_inside >= 3:
                    hand_near_face = True

                for x, y in hand_points:

                    cv2.circle(
                        image,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

            # Draw nose/mouth region.
            color = (0, 0, 255) if hand_near_face else (0, 165, 255)

            cv2.rectangle(
                image,
                (roi_x1, roi_y1),
                (roi_x2, roi_y2),
                color,
                2
            )

        # --------------------------------------------------
        # DISPLAY STATUS
        # --------------------------------------------------

        if face_count == 0:
            status = "NO FACE DETECTED"
        elif hand_near_face:
            status = "HAND NEAR NOSE/MOUTH"
        else:
            status = "FACE DETECTED - NO HAND NEAR FACE"

        cv2.putText(
            image,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255) if hand_near_face else (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"Faces: {face_count}  Hands: {hand_count}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Live Hand and Face Detection", image)

        # Press Q to stop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


except KeyboardInterrupt:
    print("Stopping receiver...")

finally:

    connection.close()
    server.close()

    face_detector.close()
    hand_detector.close()

    cv2.destroyAllWindows()

    print("Receiver stopped.")
