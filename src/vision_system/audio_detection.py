
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
from scipy.io import wavfile

# ==========================================
# ACHOO BOT - AUDIO DETECTION
# ==========================================

AUDIO_FILE = "audio/sneeze_test.wav"

# Load the YAMNet pretrained audio model
print("Loading YAMNet model...")

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

print("YAMNet loaded successfully!")

# ==========================================
# LOAD AUDIO
# ==========================================

sample_rate, audio = wavfile.read(AUDIO_FILE)

print(f"\nAudio sample rate: {sample_rate} Hz")
print(f"Audio shape: {audio.shape}")

# Convert stereo audio to mono if necessary
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Convert audio to float32 between -1 and 1
if np.issubdtype(audio.dtype, np.integer):
    audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
else:
    audio = audio.astype(np.float32)

# YAMNet requires 16 kHz audio
if sample_rate != 16000:
    raise ValueError("Audio must be sampled at 16000 Hz.")

# ==========================================
# CLASSIFY AUDIO
# ==========================================

print("\nAnalyzing audio...")

scores, embeddings, spectrogram = yamnet(
    tf.convert_to_tensor(audio, dtype=tf.float32)
)

# Average scores across audio frames
mean_scores = tf.reduce_mean(scores, axis=0).numpy()

# Get the highest score from any individual audio frame
max_scores = tf.reduce_max(scores, axis=0).numpy()

# ==========================================
# LOAD YAMNET CLASS NAMES
# ==========================================

print("\nLoading sound class names...")

class_map_path = tf.keras.utils.get_file(
    "yamnet_class_map.csv",
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
)

with open(class_map_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    class_names = [row["display_name"] for row in reader]

print("Sound class names loaded successfully!")

# ==========================================
# DISPLAY TOP PREDICTIONS
# ==========================================

top_indices = np.argsort(mean_scores)[::-1][:10]

print("\n========== TOP 10 SOUND PREDICTIONS ==========")

for index in top_indices:
    print(f"{class_names[index]}: {mean_scores[index]:.4f}")

# ==========================================
# CHECK RELEVANT SOUND CATEGORIES
# ==========================================

print("\n========== TARGET SOUND CHECK ==========")

targets = ["Sneeze", "Cough", "Clapping"]

for target in targets:
    matches = [
        i for i, name in enumerate(class_names)
        if target.lower() in name.lower()
    ]

    if matches:
        for index in matches:
            print(f"{class_names[index]}")
            print(f"Average score: {mean_scores[index]:.4f}")
            print(f"Maximum score: {max_scores[index]:.4f}")
    else:
        print(f"{target}: No matching category found.")
