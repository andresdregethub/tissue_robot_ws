
import cv2
import socket
import struct
import time


# ==========================================================
# 1. CONNECTION SETTINGS
# ==========================================================

WSL_IP = "172.20.193.245"
PORT = 9999

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 75


# ==========================================================
# 2. OPEN WINDOWS WEBCAM
# ==========================================================

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Could not open Windows webcam.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Windows webcam opened successfully.")


# ==========================================================
# 3. CONNECT TO WSL RECEIVER
# ==========================================================

connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    connection.connect((WSL_IP, PORT))
    print(f"Connected to WSL at {WSL_IP}:{PORT}")

    # ======================================================
    # 4. CAPTURE AND SEND VIDEO FRAMES
    # ======================================================

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Compress the frame into JPEG format.
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY
        ]

        success, encoded_frame = cv2.imencode(
            ".jpg",
            frame,
            encode_params
        )

        if not success:
            continue

        frame_data = encoded_frame.tobytes()

        # Send frame size first, then frame data.
        connection.sendall(
            struct.pack("!I", len(frame_data)) + frame_data
        )

        # Display the camera feed on Windows.
        cv2.imshow("Windows Webcam Sender", frame)

        # Press Q to stop sending.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except ConnectionRefusedError:
    print("Could not connect to WSL.")
    print("Make sure realtime_receiver.py is running.")

except KeyboardInterrupt:
    print("Stopping camera sender...")

finally:

    connection.close()
    camera.release()
    cv2.destroyAllWindows()

    print("Camera sender stopped.")
