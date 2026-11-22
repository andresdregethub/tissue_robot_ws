
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv

from pathlib import Path
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

# ==========================================================
# ACHOO BOT - YAMNET BATCH AUDIO EVALUATION
# ==========================================================

AUDIO_DIR = Path("audio")
OUTPUT_FILE = "audio_evaluation_results.csv"

CATEGORIES = ["sneeze", "cough", "speech", "silence"]

SAMPLE_RATE_REQUIRED = 16000

# ==========================================================
# LOAD YAMNET
# ==========================================================

print("\nLoading YAMNet model...")

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

print("YAMNet loaded successfully!")

# ==========================================================
# LOAD SOUND CLASS NAMES
# ==========================================================

print("\nLoading sound class names...")

class_map_path = tf.keras.utils.get_file(
    "yamnet_class_map.csv",
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
)

with open(class_map_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    class_names = [row["display_name"] for row in reader]

print(f"Loaded {len(class_names)} sound classes.")

# ==========================================================
# AUDIO PREPROCESSING
# ==========================================================

def prepare_audio(audio, sample_rate):

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Convert integer audio to float32
    if np.issubdtype(audio.dtype, np.integer):

        info = np.iinfo(audio.dtype)

        if info.min < 0:
            audio = audio.astype(np.float32) / max(abs(info.min), info.max)
        else:
            midpoint = (info.max + 1) / 2
            audio = (audio.astype(np.float32) - midpoint) / midpoint

    else:
        audio = audio.astype(np.float32)

    # Resample if necessary
    if sample_rate != SAMPLE_RATE_REQUIRED:

        divisor = gcd(sample_rate, SAMPLE_RATE_REQUIRED)

        up = SAMPLE_RATE_REQUIRED // divisor
        down = sample_rate // divisor

        audio = resample_poly(audio, up, down)

    # Keep values in valid audio range
    audio = np.clip(audio, -1.0, 1.0)

    return audio.astype(np.float32)


# ==========================================================
# FIND TARGET SOUND SCORES
# ==========================================================

def get_target_score(scores, target):

    matches = [
        i for i, name in enumerate(class_names)
        if target.lower() in name.lower()
    ]

    if not matches:
        return 0.0, 0.0

    average_score = max(
        float(np.mean(scores[:, i]))
        for i in matches
    )

    maximum_score = max(
        float(np.max(scores[:, i]))
        for i in matches
    )

    return average_score, maximum_score


# ==========================================================
# PROCESS DATASET
# ==========================================================

results = []

print("\nStarting dataset evaluation...\n")

for category in CATEGORIES:

    category_folder = AUDIO_DIR / category

    if not category_folder.exists():
        print(f"WARNING: Missing folder: {category_folder}")
        continue

    audio_files = sorted(category_folder.glob("*.wav"))

    print(f"\nProcessing {category.upper()} ({len(audio_files)} files)")

    for audio_file in audio_files:

        try:
            sample_rate, audio = wavfile.read(audio_file)

            audio = prepare_audio(audio, sample_rate)

            # Run YAMNet
            scores, embeddings, spectrogram = yamnet(
                tf.convert_to_tensor(audio, dtype=tf.float32)
            )

            scores = scores.numpy()

            # Average scores across frames
            mean_scores = np.mean(scores, axis=0)

            # Maximum score from any frame
            max_scores = np.max(scores, axis=0)

            # Top five average predictions
            top_indices = np.argsort(mean_scores)[::-1][:5]

            top_predictions = [
                f"{class_names[i]} ({mean_scores[i]:.4f})"
                for i in top_indices
            ]

            # Target sound scores
            sneeze_avg, sneeze_max = get_target_score(scores, "Sneeze")
            cough_avg, cough_max = get_target_score(scores, "Cough")
            speech_avg, speech_max = get_target_score(scores, "Speech")
            silence_avg, silence_max = get_target_score(scores, "Silence")

            predicted_class = class_names[top_indices[0]]

            result = {
                "filename": audio_file.name,
                "actual_category": category,
                "top_prediction": predicted_class,
                "top_score": float(mean_scores[top_indices[0]]),

                "sneeze_average": sneeze_avg,
                "sneeze_maximum": sneeze_max,

                "cough_average": cough_avg,
                "cough_maximum": cough_max,

                "speech_average": speech_avg,
                "speech_maximum": speech_max,

                "silence_average": silence_avg,
                "silence_maximum": silence_max,

                "top_5_predictions": " | ".join(top_predictions)
            }

            results.append(result)

            print(f"\nFile: {audio_file.name}")
            print(f"Actual category: {category}")
            print(f"Top prediction: {predicted_class}")

            print(f"Sneeze max: {sneeze_max:.4f}")
            print(f"Cough max: {cough_max:.4f}")

        except Exception as error:
            print(f"\nERROR processing {audio_file}: {error}")


# ==========================================================
# EXPORT RESULTS TO CSV
# ==========================================================

if results:

    fieldnames = list(results[0].keys())

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print("\n========================================")
    print("BATCH EVALUATION COMPLETE")
    print("========================================")

    print(f"Successfully processed: {len(results)} files")
    print(f"Results saved to: {OUTPUT_FILE}")

else:
    print("\nNo recordings were successfully processed.")

print("\nAudio evaluation finished!")
