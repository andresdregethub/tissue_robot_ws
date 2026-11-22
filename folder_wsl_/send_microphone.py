
import socket
import sounddevice as sd
import time

# ==========================================================
# SETTINGS
# ==========================================================

# Replace this with the IP address from hostname -I in Ubuntu.
UBUNTU_IP = "172.20.193.245"

PORT = 9998

DEVICE = 1
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


# ==========================================================
# CONNECT TO UBUNTU
# ==========================================================

print("Connecting to Ubuntu audio receiver...")

try:
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.connect((UBUNTU_IP, PORT))

    print("Connected successfully!")
    print("Starting microphone...")
    print("Press CTRL+C to stop.\n")

    # ======================================================
    # CAPTURE AND SEND MICROPHONE AUDIO
    # ======================================================

    with sd.InputStream(
        device=DEVICE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SAMPLES
    ) as microphone:

        while True:
            audio, overflowed = microphone.read(CHUNK_SAMPLES)

            if overflowed:
                print("Warning: microphone buffer overflow")

            connection.sendall(audio.tobytes())

except KeyboardInterrupt:
    print("\nStopping microphone sender.")

except Exception as error:
    print(f"\nError: {error}")

finally:
    if "connection" in locals():
        connection.close()

    print("Microphone sender stopped.")
