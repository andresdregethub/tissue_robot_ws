
import socket
import csv
import time
from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# ==========================================================
# SETTINGS
# ==========================================================

HOST = "0.0.0.0"
PORT = 9998

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
CHUNK_BYTES = CHUNK_SAMPLES * 2

BUFFER_SECONDS = 1.5
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_SECONDS)

SNEEZE_THRESHOLD = 0.75
COOLDOWN_SECONDS = 5

last_sneeze_time = 0


# ==========================================================
# LOAD YAMNET
# ==========================================================

print("Loading YAMNet model...")

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

print("YAMNet loaded successfully!")

class_map_path = tf.keras.utils.get_file(
    "yamnet_class_map.csv",
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
)

with open(class_map_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    class_names = [row["display_name"] for row in reader]

sneeze_index = class_names.index("Sneeze")

print("Sneeze class found successfully!")


# ==========================================================
# RECEIVE EXACT NUMBER OF BYTES
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
# EVALUATE AUDIO
# ==========================================================

def evaluate_audio(audio_buffer):

    global last_sneeze_time

    audio = np.asarray(audio_buffer, dtype=np.float32)

    scores, embeddings, spectrogram = yamnet(
        tf.convert_to_tensor(audio, dtype=tf.float32)
    )

    frame_scores = scores.numpy()

    # Average scores across the evaluated audio window.
    mean_scores = np.mean(frame_scores, axis=0)

    top_index = int(np.argmax(mean_scores))
    top_name = class_names[top_index]
    top_score = float(mean_scores[top_index])

    # Strongest Sneeze score across all frames.
    sneeze_max = float(np.max(frame_scores[:, sneeze_index]))

    # Find the strongest individual frame.
    best_frame = int(np.argmax(np.max(frame_scores, axis=1)))
    frame_top_index = int(np.argmax(frame_scores[best_frame]))

    frame_top_name = class_names[frame_top_index]
    frame_top_score = float(frame_scores[best_frame, frame_top_index])

    print("\n========== LIVE AUDIO RESULTS ==========")
    print(f"Overall prediction: {top_name} ({top_score:.4f})")
    print(f"Strongest frame:    {frame_top_name} ({frame_top_score:.4f})")
    print(f"Sneeze max score:   {sneeze_max:.4f}")

    # ======================================================
    # SNEEZE EVENT DETECTION
    # ======================================================

    current_time = time.monotonic()

    if sneeze_max >= SNEEZE_THRESHOLD:

        if current_time - last_sneeze_time >= COOLDOWN_SECONDS:

            print("\n*** SNEEZE EVENT DETECTED ***")
            print(f"Sneeze confidence score: {sneeze_max:.4f}")

            last_sneeze_time = current_time

        else:
            print("Sneeze signal detected, but cooldown is active.")

    print("========================================")


# ==========================================================
# START SERVER
# ==========================================================

print("\nStarting live audio receiver...")
print(f"Listening on port {PORT}")
print("Waiting for Windows microphone connection...\n")

audio_buffer = deque(maxlen=BUFFER_SAMPLES)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:

    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    connection, address = server.accept()

    with connection:

        print(f"Connected to microphone sender: {address}")

        try:

            while True:

                packet = receive_exactly(connection, CHUNK_BYTES)

                if packet is None:
                    print("Microphone connection closed.")
                    break

                # Convert incoming audio into float32.
                audio_chunk = np.frombuffer(
                    packet,
                    dtype=np.int16
                ).astype(np.float32) / 32768.0

                audio_buffer.extend(audio_chunk)

                # Evaluate once the rolling buffer is full.
                if len(audio_buffer) == BUFFER_SAMPLES:
                    evaluate_audio(audio_buffer)

        except KeyboardInterrupt:
            print("\nStopping live audio receiver.")

print("Receiver stopped.")
