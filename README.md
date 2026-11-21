click the link to see the test running : https://imgur.com/a/DKhUKVd


# tissue_robot_ws
# Vision System – Autonomous Tissue Delivery Robot

## Overview

This project is part of a simulated autonomous tissue-delivery robot developed using ROS 2 Jazzy and Gazebo Harmonic.

The vision system uses OpenCV and MediaPipe to detect human faces and hands. Its current objective is to identify when a person's hand moves near their nose or mouth, which may be useful as one signal for recognizing a possible sneeze or request for assistance.

The system currently supports:

- Human face detection.
- Hand landmark detection.
- Hand-to-face proximity detection.
- Static image processing.
- Real-time webcam processing.
- Video transmission from Windows to Ubuntu through TCP.

**Note:** Hand proximity is currently a basic detection heuristic. It does not definitively identify sneezing or determine whether someone needs a tissue.

---

## 1. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3 | Main programming language |
| OpenCV | Image and video processing |
| MediaPipe | Face and hand landmark detection |
| NumPy | Image array processing |
| Socket / TCP | Video transmission between Windows and WSL |
| Ubuntu 24.04 | Linux development environment |
| WSL 2 | Runs Ubuntu on Windows |
| VS Code | Code editor |

### Tested Environment

The original development environment:

- Operating System: Windows 11
- Linux Environment: Ubuntu 24.04 LTS
- Architecture: AMD64 / x86_64
- WSL Version: WSL 2
- Python: Ubuntu system Python (3.12)
- ROS 2: Jazzy
- Gazebo: Harmonic

The vision system can be developed independently of ROS 2 and Gazebo.

---

## 2. Project Structure

```text
vision_system/
│
├── README.md
├── requirements.txt
│
├── face_detection.py
├── hand_face_detection.py
├── realtime_receiver.py
├── camera_sender.py
├── download_image.py
│
├── models/
│   ├── face_landmarker.task
│   └── hand_landmarker.task
│
├── images/
│   └── test images
│
└── .gitignore
```

### Python Files

| File | Description |
|---|---|
| `face_detection.py` | Detects faces in a static image. |
| `hand_face_detection.py` | Detects faces and hands and checks whether hand landmarks overlap the estimated nose/mouth region. |
| `realtime_receiver.py` | Receives video frames from Windows and performs real-time MediaPipe detection. |
| `camera_sender.py` | Captures webcam frames on Windows and transmits them to Ubuntu. |
| `download_image.py` | Helper script for downloading test images. |

---

## 3. Requirements

### Ubuntu / WSL Requirements

Install the following packages:

```bash
sudo apt update

sudo apt install -y \
python3 \
python3-pip \
python3-venv \
python3-dev \
build-essential \
libgl1 \
libglib2.0-0
```

### Python Dependencies

Create a file called `requirements.txt`:

```text
numpy==1.26.4
opencv-python==4.11.0.86
mediapipe==1.2.3
```

These are the versions used in the current WSL development environment.

---

## 4. Environment Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd vision_system
```

### Step 2: Create a Python Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install requirements:

```bash
python -m pip install -r requirements.txt
```

Verify the installed libraries:

```bash
python -c "import cv2, mediapipe, numpy; print('OpenCV:', cv2.__version__); print('MediaPipe:', mediapipe.__version__); print('NumPy:', numpy.__version__)"
```

Expected versions:

```text
OpenCV: 4.11.0
MediaPipe: 1.2.3
NumPy: 1.26.4
```

---

## 5. Download MediaPipe Models

The vision system requires two pretrained MediaPipe models.

Create the model directory:

```bash
mkdir -p models
```

Download the Face Landmarker:

```bash
wget -O models/face_landmarker.task \
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

Download the Hand Landmarker:

```bash
wget -O models/hand_landmarker.task \
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Verify that both models exist:

```bash
ls -lh models/
```

Expected files:

```text
face_landmarker.task
hand_landmarker.task
```

---

## 6. Running Static Image Detection

### Face Detection

Make sure a test image is located in the `images/` directory.

Run:

```bash
python face_detection.py
```

The script detects faces and saves an annotated image.

### Hand-to-Face Detection

Run:

```bash
python hand_face_detection.py
```

This script:

1. Loads a test image.
2. Detects faces.
3. Detects hands.
4. Estimates the nose and mouth region.
5. Checks for hand landmark overlap.
6. Saves the annotated result.

The output image is:

```text
images/hand_face_annotated.jpg
```

---

## 7. Real-Time Webcam Detection

The real-time implementation uses two programs because the webcam is accessed through Windows while MediaPipe runs in Ubuntu/WSL.

### System Architecture

```text
Windows 11
    |
    v
Integrated Webcam
    |
    v
camera_sender.py
    |
    | TCP video transmission
    |
    v
Ubuntu 24.04 / WSL 2
    |
    v
realtime_receiver.py
    |
    v
OpenCV + MediaPipe
    |
    v
Face and Hand Detection
```

### Windows Requirements

Install Python and OpenCV on Windows.

In PowerShell:

```powershell
py -m pip install opencv-python
```

Check the installation:

```powershell
py -c "import cv2; print(cv2.__version__)"
```

### Step 1: Configure the WSL IP Address

Inside Ubuntu:

```bash
hostname -I
```

Copy the WSL IP address.

Open `camera_sender.py` on Windows and update:

```python
WSL_IP = "YOUR_WSL_IP_ADDRESS"
PORT = 9999
```

Replace the example address with the IP address returned by Ubuntu.

### Step 2: Start the Ubuntu Receiver

Inside Ubuntu:

```bash
cd ~/vision_system
source .venv/bin/activate
python realtime_receiver.py
```

Expected message:

```text
Waiting for Windows camera on port 9999...
Start the Windows sender to connect.
```

### Step 3: Start the Windows Camera Sender

Open PowerShell:

```powershell
py camera_sender.py
```

Expected output:

```text
Windows webcam opened successfully.
Connected to WSL at YOUR_WSL_IP:9999
```

### Step 4: Test Real-Time Detection

The system should display:

- Windows webcam preview.
- Ubuntu live detection window.
- Face count.
- Hand count.
- Hand-near-nose/mouth status.

Press `Q` in the preview window to stop the corresponding program.

**Networking note:** If Windows cannot connect to WSL, check the current WSL IP address, Windows Firewall rules, and WSL networking configuration. The IP address can change after restarting WSL.

---

## 8. Current Detection Logic

The current hand-to-face detector uses MediaPipe landmarks.

It estimates the nose and mouth region using the detected face boundaries.

It then checks whether at least three hand landmarks fall inside that estimated region.

Current output states:

- `NO FACE DETECTED`
- `NO HAND COVER DETECTED`
- `HAND NEAR NOSE/MOUTH`

This is an initial proximity-based implementation and should not be treated as a reliable sneeze classifier.

---

## 9. Current Development Status

| Feature | Status |
|---|---|
| Face detection | Working |
| Hand detection | Working |
| Hand-to-face proximity | Working |
| Static image processing | Working |
| Windows webcam access | Working |
| Windows-to-WSL video stream | Working |
| Real-time detection | Working |
| Sneeze audio classification | Planned |
| Improved gesture recognition | Planned |
| ROS 2 integration | Planned |
| Gazebo robot integration | Planned |

---

## 10. Future Development

Planned improvements include:

1. Improve hand-to-face detection accuracy.
2. Track hand movement across multiple frames.
3. Add real-time sneeze audio classification.
4. Combine visual and audio detection.
5. Create a ROS 2 vision node.
6. Publish person and sneeze-related detection topics.
7. Connect the vision system to the simulated tissue-delivery robot.

---

## 11. Important Notes

- The Windows camera sender and Ubuntu receiver must run simultaneously.
- The webcam must be accessible to Windows.
- Both MediaPipe model files must exist in the `models/` directory.
- The WSL IP address must match the address configured in the Windows sender.
- Python dependencies should be installed inside the Ubuntu virtual environment.
- The Windows sender requires its own Windows Python installation and OpenCV.
- Do not commit `.venv/`, Python cache files, or generated output images unnecessarily.
- Check model and dataset licenses before redistributing them.

