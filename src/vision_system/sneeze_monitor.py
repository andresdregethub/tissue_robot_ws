import csv
import socket
import struct
import threading
import time
import rclpy
from collections import deque
from std_msgs.msg import Bool, Float32

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================================
# 1. SETTINGS
# ==========================================================

HOST = "0.0.0.0"

CAMERA_PORT = 9999
AUDIO_PORT = 9998

FACE_MODEL_PATH = "models/face_landmarker.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
CHUNK_BYTES = CHUNK_SAMPLES * 2

BUFFER_SECONDS = 1.5
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_SECONDS)

SNEEZE_THRESHOLD = 0.75
COOLDOWN_SECONDS = 5
AUDIO_EVENT_WINDOW = 4

# Shared audio information.
state_lock = threading.Lock()

audio_sneeze_score = 0.0
audio_event_until = 0.0
audio_event_id = 0


# ==========================================================
# 2. NETWORK HELPER
# ==========================================================

def receive_exactly(connection, amount):
    data = bytearray()

    while len(data) < amount:
        packet = connection.recv(amount - len(data))

        if not packet:
            return None

        data.extend(packet)

    return bytes(data)


# ==========================================================
# 3. AUDIO RECEIVER THREAD
# ==========================================================

def audio_receiver():

    global audio_sneeze_score
    global audio_event_until
    global audio_event_id

    print("\n[AUDIO] Loading YAMNet...")

    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    class_map_path = tf.keras.utils.get_file(
        "yamnet_class_map.csv",
        "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    )

    with open(class_map_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        class_names = [row["display_name"] for row in reader]

    sneeze_index = class_names.index("Sneeze")

    print("[AUDIO] YAMNet loaded successfully.")
    print(f"[AUDIO] Listening on port {AUDIO_PORT}...")

    audio_buffer = deque(maxlen=BUFFER_SAMPLES)
    last_sneeze_time = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:

        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, AUDIO_PORT))
        server.listen(1)

        connection, address = server.accept()

        with connection:

            print(f"[AUDIO] Connected to microphone: {address}")

            while True:

                packet = receive_exactly(connection, CHUNK_BYTES)

                if packet is None:
                    print("[AUDIO] Microphone disconnected.")
                    break

                audio_chunk = np.frombuffer(
                    packet,
                    dtype=np.int16
                ).astype(np.float32) / 32768.0

                audio_buffer.extend(audio_chunk)

                if len(audio_buffer) < BUFFER_SAMPLES:
                    continue

                audio = np.asarray(audio_buffer, dtype=np.float32)

                scores, embeddings, spectrogram = yamnet(
                    tf.convert_to_tensor(audio, dtype=tf.float32)
                )

                frame_scores = scores.numpy()

                sneeze_max = float(
                    np.max(frame_scores[:, sneeze_index])
                )

                mean_scores = np.mean(frame_scores, axis=0)
                top_index = int(np.argmax(mean_scores))

                print(
                    f"[AUDIO] Prediction: {class_names[top_index]} | "
                    f"Sneeze score: {sneeze_max:.4f}"
                )

                current_time = time.monotonic()

                with state_lock:
                    audio_sneeze_score = sneeze_max

                    if (
                        sneeze_max >= SNEEZE_THRESHOLD
                        and current_time - last_sneeze_time >= COOLDOWN_SECONDS
                    ):
                        last_sneeze_time = current_time
                        audio_event_until = (
                            current_time + AUDIO_EVENT_WINDOW
                        )
                        audio_event_id += 1

                        print("\n[AUDIO] *** SNEEZE EVENT DETECTED ***")
                        print(
                            f"[AUDIO] Sneeze score: {sneeze_max:.4f}\n"
                        )


# ==========================================================
# 4. INITIALIZE MEDIAPIPE
# ==========================================================

def create_detectors():

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

    return face_detector, hand_detector


# ==========================================================
# 5. START CAMERA RECEIVER
# ==========================================================

def camera_receiver(ros_node, ros_publishers):

    global audio_sneeze_score
    global audio_event_until
    global audio_event_id

    face_detector, hand_detector = create_detectors()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((HOST, CAMERA_PORT))
    server.listen(1)

    print(f"\n[CAMERA] Listening on port {CAMERA_PORT}...")
    print("[CAMERA] Waiting for Windows webcam sender...")

    connection, address = server.accept()

    print(f"[CAMERA] Connected to webcam: {address}")

    previous_timestamp = 0
    reported_event_id = 0
    last_ros_publish = 0.0
    ROS_PUBLISH_INTERVAL = 0.2

    try:

        with connection:

            while True:

                size_data = receive_exactly(connection, 4)

                if size_data is None:
                    print("[CAMERA] Webcam disconnected.")
                    break

                frame_size = struct.unpack("!I", size_data)[0]

                if frame_size <= 0 or frame_size > 10_000_000:
                    raise ValueError("Invalid video frame size")

                frame_data = receive_exactly(connection, frame_size)

                if frame_data is None:
                    break

                encoded = np.frombuffer(
                    frame_data,
                    dtype=np.uint8
                )

                image = cv2.imdecode(
                    encoded,
                    cv2.IMREAD_COLOR
                )

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

                timestamp = max(
                    int(time.monotonic() * 1000),
                    previous_timestamp + 1
                )

                previous_timestamp = timestamp

                faces = face_detector.detect_for_video(
                    mp_image,
                    timestamp
                )

                hands = hand_detector.detect_for_video(
                    mp_image,
                    timestamp
                )

                face_count = len(faces.face_landmarks)
                hand_count = len(hands.hand_landmarks)

                hand_near_face = False

                # --------------------------------------------------
                # FIND NOSE/MOUTH REGION
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

                    roi_color = (
                        (0, 0, 255)
                        if hand_near_face
                        else (0, 165, 255)
                    )

                    cv2.rectangle(
                        image,
                        (roi_x1, roi_y1),
                        (roi_x2, roi_y2),
                        roi_color,
                        2
                    )

                # --------------------------------------------------
                # READ AUDIO DETECTION STATE
                # --------------------------------------------------

                current_time = time.monotonic()

                with state_lock:
                    current_score = audio_sneeze_score
                    current_event_until = audio_event_until
                    current_event_id = audio_event_id

                recent_sneeze = current_time <= current_event_until

                # --------------------------------------------------
                # COMBINE AUDIO AND VISUAL CUES
                # --------------------------------------------------

                combined_detection = (
                    recent_sneeze and hand_near_face
                )
                                # --------------------------------------------------
                # PUBLISH PERCEPTION DATA TO ROS 2
                # --------------------------------------------------

                if current_time - last_ros_publish >= ROS_PUBLISH_INTERVAL:

                    ros_publishers["face_detected"].publish(
                        Bool(data=(face_count > 0))
                    )

                    ros_publishers["hand_near_face"].publish(
                        Bool(data=hand_near_face)
                    )

                    ros_publishers["sneeze_detected"].publish(
                        Bool(data=combined_detection)
                    )

                    ros_publishers["sneeze_score"].publish(
                        Float32(data=float(current_score))
                    )

                    last_ros_publish = current_time

                    # Process ROS events without blocking the camera loop.
                    rclpy.spin_once(
                        ros_node,
                        timeout_sec=0.0
                    )

                if (
                    combined_detection
                    and current_event_id != reported_event_id
                ):

                    print("\n========================================")
                    print("*** COMBINED SNEEZE CUES DETECTED ***")
                    print("Audio: Sneeze event detected")
                    print("Vision: Hand near nose/mouth")
                    print("Achoo Bot: Possible sneeze event")
                    print("========================================\n")

                    reported_event_id = current_event_id

                # --------------------------------------------------
                # DISPLAY STATUS
                # --------------------------------------------------

                if face_count == 0:
                    vision_status = "NO FACE DETECTED"
                elif hand_near_face:
                    vision_status = "HAND NEAR NOSE/MOUTH"
                else:
                    vision_status = "FACE DETECTED - NO HAND NEAR FACE"

                audio_status = (
                    "SNEEZE SIGNAL ACTIVE"
                    if recent_sneeze
                    else "NO RECENT SNEEZE SIGNAL"
                )

                if combined_detection:
                    overall_status = "POSSIBLE SNEEZE EVENT"
                    status_color = (0, 0, 255)
                else:
                    overall_status = "MONITORING"
                    status_color = (0, 255, 0)

                cv2.putText(
                    image,
                    vision_status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    image,
                    audio_status,
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"Sneeze score: {current_score:.3f}",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    image,
                    overall_status,
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2
                )

                cv2.imshow("Achoo Bot - Audio and Vision Monitor", image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:

        connection.close()
        server.close()

        face_detector.close()
        hand_detector.close()

        cv2.destroyAllWindows()

        print("[CAMERA] Camera receiver stopped.")


# ==========================================================
# 6. MAIN PROGRAM
# ==========================================================

if __name__ == "__main__":

    print("========================================")
    print("       ACHOO BOT - SNEEZE MONITOR")
    print("========================================")

    # ------------------------------------------------------
    # INITIALIZE ROS 2
    # ------------------------------------------------------

    rclpy.init()

    ros_node = rclpy.create_node("achoobot_perception")

    ros_publishers = {

        "face_detected": ros_node.create_publisher(
            Bool,
            "/perception/face_detected",
            10
        ),

        "hand_near_face": ros_node.create_publisher(
            Bool,
            "/perception/hand_near_face",
            10
        ),

        "sneeze_detected": ros_node.create_publisher(
            Bool,
            "/perception/sneeze_detected",
            10
        ),

        "sneeze_score": ros_node.create_publisher(
            Float32,
            "/perception/sneeze_score",
            10
        )

    }

    print("[ROS 2] Perception node initialized.")
    print("[ROS 2] Publishers created.")

    # ------------------------------------------------------
    # START AUDIO RECEIVER
    # ------------------------------------------------------

    audio_thread = threading.Thread(
        target=audio_receiver,
        daemon=True
    )

    audio_thread.start()

    # ------------------------------------------------------
    # START CAMERA RECEIVER
    # ------------------------------------------------------

    try:

        camera_receiver(
            ros_node,
            ros_publishers
        )

    except KeyboardInterrupt:

        print("\nStopping Achoo Bot monitor...")

    finally:

        ros_node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print("[ROS 2] Perception node shut down.")
        print("Monitor stopped.")
