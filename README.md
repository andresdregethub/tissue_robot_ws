# Achoo Bot – Vision and Perception System
## Autonomous Tissue Delivery Robot | ROS 2 Jazzy

## Overview

This project is part of a simulated autonomous tissue-delivery robot developed using **ROS 2 Jazzy and Gazebo Harmonic**.

The perception system detects faces, hands, hand-to-face proximity, and possible sneeze events using computer vision and audio classification.

The system combines:
- OpenCV and MediaPipe for face and hand detection.
- YAMNet for audio classification and sneeze-event detection.
- TCP communication to transmit webcam and microphone data from Windows to Ubuntu WSL.
- ROS 2 publishers to communicate perception results to other robot components.

The main objective is to detect a possible sneeze event and publish a ROS 2 trigger that another subsystem can use to initiate the tissue-delivery sequence.

**Important:** This project is simulation-first. The perception system detects events and publishes triggers. Robot movement, navigation, and tissue manipulation are handled by other robot subsystems.

## 1. Current Features

### Computer Vision
- Human face detection.
- Hand landmark detection.
- Hand-to-face proximity detection.
- Static image processing.
- Real-time webcam processing.
- Windows-to-Ubuntu video transmission through TCP.

### Audio Detection
- Real-time microphone audio streaming.
- YAMNet audio classification.
- Sneeze, cough, speech, and silence evaluation.
- Sneeze-score monitoring.
- Audio event cooldown and detection window.

### Combined Perception
- Combines audio sneeze detection with hand-near-face detection.
- Publishes a combined sneeze event.
- Provides detection results through ROS 2 topics.

### ROS 2 Integration
- ROS 2 perception node.
- Four ROS 2 publishers.
- Registered ROS 2 executable.
- Independent topic monitoring through ROS 2 CLI.

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Main programming language |
| OpenCV | Image and video processing |
| MediaPipe | Face and hand landmark detection |
| NumPy | Image and audio array processing |
| TensorFlow | Audio classification support |
| TensorFlow Hub | Loading pretrained YAMNet |
| YAMNet | Audio event classification |
| Socket / TCP | Windows-to-WSL data transmission |
| ROS 2 Jazzy | Robot communication and perception topics |
| Gazebo Harmonic | Robot simulation environment |
| Ubuntu 24.04 | Linux development environment |
| WSL 2 | Runs Ubuntu on Windows |

## 3. Tested Environment

- Operating System: Windows 11
- Linux Environment: Ubuntu 24.04 LTS
- Architecture: AMD64 / x86_64
- WSL Version: WSL 2
- Python: 3.12
- ROS 2: Jazzy
- Gazebo: Harmonic
- Gazebo Sim: 8.15.0

The perception system can be developed and tested independently of Gazebo.

## 4. Project Structure

The relevant workspace structure is:

```text
tissue_robot_ws/
│
├── src/
│   │
│   ├── vision_system/
│   │   ├── sneeze_monitor.py
│   │   ├── hand_face_detection.py
│   │   ├── face_detection.py
│   │   ├── realtime_receiver.py
│   │   ├── live_audio_receiver.py
│   │   ├── audio_detection.py
│   │   ├── evaluate_audio_dataset.py
│   │   ├── download_image.py
│   │   ├── requirements.txt
│   │   │
│   │   ├── models/
│   │   │   ├── face_landmarker.task
│   │   │   └── hand_landmarker.task
│   │   │
│   │   ├── images/
│   │   │
│   │   └── audio/
│   │
│   └── tissue_robot/
│       ├── package.xml
│       ├── setup.py
│       ├── setup.cfg
│       ├── resource/
│       │   └── tissue_robot
│       ├── tissue_robot/
│       │   ├── __init__.py
│       │   └── sneeze_monitor_launcher.py
│       └── test/
│
├── build/                 # Generated; do not commit
├── install/               # Generated; do not commit
├── log/                   # Generated; do not commit
└── README.md
```

Windows sender scripts are maintained separately and must also be provided to teammates:

- Webcam sender Python script.
- Microphone sender Python script.

Their filenames and locations should be documented in the project setup instructions.

## 5. Python Dependencies

The vision system uses the following Python dependencies:

```text
numpy==1.26.4
opencv-python==4.11.0.86
mediapipe==1.2.3
tensorflow==2.16.2
tensorflow-hub==0.16.1
sounddevice
scipy
```

ROS 2 Python dependencies are supplied by the ROS 2 installation, including:

- rclpy
- std_msgs

Do not install ROS 2 Python packages by replacing the Ubuntu ROS installation with unrelated pip versions.

## 6. Environment Setup

### Step 1: Install Ubuntu Requirements

Inside Ubuntu WSL:

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

Make sure ROS 2 Jazzy is installed and configured.

### Step 2: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL

cd tissue_robot_ws
```

### Step 3: Create the Python Virtual Environment

```bash
cd src/vision_system

python3 -m venv .venv

source .venv/bin/activate
```

### Step 4: Install Python Dependencies

```bash
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

Source ROS 2:

```bash
source /opt/ros/jazzy/setup.bash
```

### Step 5: Download MediaPipe Models

The vision system requires two pretrained MediaPipe models.

Create the models directory:

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

Verify:

```bash
ls -lh models/
```

Both model files must exist before running the perception system.

## 7. Windows Camera and Microphone Streaming

WSL does not directly access the Windows webcam and microphone in the current setup.

Instead, Windows sender scripts capture the devices and transmit their data to Ubuntu through TCP.

### System Architecture

```text
WINDOWS 11
    |
    ├── Webcam
    |     |
    |     └── Camera Sender
    |
    └── Microphone
          |
          └── Microphone Sender
                 |
                 | TCP Transmission
                 |
                 v
UBUNTU 24.04 / WSL 2
    |
    ├── Camera Receiver
    |
    └── Audio Receiver
           |
           v
    Sneeze Monitor
           |
           v
    OpenCV + MediaPipe + YAMNet
           |
           v
    ROS 2 Perception Publishers
```

### Windows Requirements

Install Python and the dependencies required by the sender scripts.

For the webcam sender:

```powershell
py -m pip install opencv-python
```

For the microphone sender:

```powershell
py -m pip install sounddevice numpy
```

### Configure the WSL IP Address

Inside Ubuntu:

```bash
hostname -I
```

Use the appropriate WSL IP address in the Windows sender scripts.

The current TCP ports are:

| Stream | Port |
|---|---|
| Webcam | 9999 |
| Microphone | 9998 |

The Windows sender scripts and Ubuntu receivers must use matching IP addresses and ports.

The WSL IP address may change after restarting WSL.

## 8. Audio Detection and Evaluation

The audio system uses Google's pretrained YAMNet model through TensorFlow Hub.

The system evaluates audio and calculates a score for the sneeze sound class.

Current configuration includes:

- Sample rate: 16,000 Hz
- Audio chunk duration: 0.5 seconds
- Audio buffer: 1.5 seconds
- Sneeze threshold: 0.75
- Cooldown period: 5 seconds
- Audio event window: 4 seconds

The project also includes an audio evaluation dataset organized around four categories:

- Sneeze
- Cough
- Speech
- Silence

The evaluation script is:

```bash
python evaluate_audio_dataset.py
```

The evaluation results are saved in:

```text
audio_evaluation_results.csv
```

**Note:** YAMNet is a pretrained general audio classifier. The current evaluation process does not fine-tune or retrain the model. Its sneeze score is a classification score, not a calibrated probability.

## 9. Combined Sneeze Detection Logic

The current perception system combines audio and visual information.

The combined event is activated when:

1. YAMNet detects a sneeze-related audio event above the configured threshold.
2. MediaPipe detects a hand near the person's nose or mouth.
3. The audio event is still within the configured event window.

The combined detection is:

```python
combined_detection = recent_sneeze and hand_near_face
```

This reduces reliance on audio or visual detection alone.

It is still an initial heuristic and does not guarantee that a person actually sneezed or needs a tissue.

## 10. ROS 2 Integration

The perception system is integrated with ROS 2 Jazzy.

The main perception script is:

```text
vision_system/sneeze_monitor.py
```

It creates a ROS 2 node named:

```text
achoobot_perception
```

The node publishes four topics.

### ROS 2 Topics

| Topic | Message Type | Description |
|---|---|---|
| `/perception/face_detected` | `std_msgs/msg/Bool` | Indicates whether a face is detected |
| `/perception/hand_near_face` | `std_msgs/msg/Bool` | Indicates whether a hand is near the nose or mouth |
| `/perception/sneeze_detected` | `std_msgs/msg/Bool` | Combined audio and visual sneeze event |
| `/perception/sneeze_score` | `std_msgs/msg/Float32` | Current YAMNet sneeze-class score |

### Topic Descriptions

**face_detected**

Publishes `true` when at least one face is detected.

**hand_near_face**

Publishes `true` when the hand-to-face proximity heuristic is satisfied.

**sneeze_detected**

Publishes `true` when the combined audio and visual detection conditions are satisfied. This is the primary trigger intended for the robot behavior subsystem.

**sneeze_score**

Publishes the current YAMNet sneeze-class score.

### ROS 2 Executable

The perception system is registered as a ROS 2 executable through the `tissue_robot` package.

The launcher file is:

```text
src/tissue_robot/tissue_robot/sneeze_monitor_launcher.py
```

The executable entry point in `setup.py` is:

```python
entry_points={
    'console_scripts': [
        'sneeze_monitor = tissue_robot.sneeze_monitor_launcher:main',
    ],
},
```

The launcher starts the existing perception script without requiring the original detection logic to be moved into the ROS 2 package.

It also sets the working directory to `vision_system` so that the MediaPipe model paths resolve correctly.

### Build the ROS 2 Package

From the workspace root:

```bash
cd ~/tissue_robot_ws

source /opt/ros/jazzy/setup.bash

colcon build --symlink-install --packages-select tissue_robot
```

Source the workspace:

```bash
source install/setup.bash
```

Verify the executable:

```bash
ros2 pkg executables tissue_robot
```

Expected output:

```text
tissue_robot sneeze_monitor
```

### Run the Perception System

Start the Windows webcam sender and microphone sender first.

Then, inside Ubuntu:

```bash
cd ~/tissue_robot_ws

source /opt/ros/jazzy/setup.bash
source src/vision_system/.venv/bin/activate
source install/setup.bash

ros2 run tissue_robot sneeze_monitor
```

The system should initialize YAMNet, start listening on TCP ports 9998 and 9999, and connect to the Windows senders.

### Verify ROS 2 Topics

Open a second Ubuntu terminal:

```bash
source /opt/ros/jazzy/setup.bash

cd ~/tissue_robot_ws

source install/setup.bash
```

List the topics:

```bash
ros2 topic list
```

Expected perception topics:

```text
/perception/face_detected
/perception/hand_near_face
/perception/sneeze_detected
/perception/sneeze_score
```

Monitor individual topics:

```bash
ros2 topic echo /perception/face_detected
```

```bash
ros2 topic echo /perception/hand_near_face
```

```bash
ros2 topic echo /perception/sneeze_detected
```

```bash
ros2 topic echo /perception/sneeze_score
```

Each command continuously displays messages until stopped with `Ctrl+C`.

## 11. Current Development Status

| Feature | Status |
|---|---|
| Face detection | Working |
| Hand detection | Working |
| Hand-to-face proximity detection | Working |
| Static image processing | Working |
| Windows webcam access | Working |
| Windows-to-WSL video transmission | Working |
| Real-time vision processing | Working |
| YAMNet audio classification | Working |
| Audio dataset evaluation | Working |
| Combined audio and visual sneeze detection | Working |
| ROS 2 perception node | Working |
| Four ROS 2 publishers | Working |
| ROS 2 executable registration | Working |
| ROS 2 executable launch | Tested working |
| Independent ROS 2 topic monitoring | Verified |
| Robot movement trigger integration | Pending |
| Gazebo robot behavior integration | Pending |

## 12. Project Responsibilities and Integration

The perception subsystem is responsible for identifying possible sneeze events and publishing ROS 2 messages.

The robot behavior subsystem can subscribe to:

```text
/perception/sneeze_detected
```

When it receives `true`, it can initiate the appropriate tissue-delivery behavior.

The perception subsystem does not control:

- Robot navigation.
- Arm movement.
- Gripper movement.
- Tissue localization.
- Tissue grasping.
- Tissue delivery.

These functions belong to other robot subsystems.

## 13. Important Notes

- Start both Windows sender scripts before launching the full perception system.
- The Windows webcam and microphone must be accessible to their respective sender scripts.
- The MediaPipe model files must exist in `vision_system/models/`.
- The WSL IP address must match the address configured in the Windows senders.
- TCP ports 9998 and 9999 must be available.
- Source ROS 2 Jazzy before running ROS 2 commands.
- Activate the appropriate Python virtual environment before running the perception executable.
- Do not upload `.venv/`, `__pycache__/`, `.pyc` files, or generated build/install/log directories.
- Check the licenses of model files, datasets, and test images before redistributing them.

## 14. Future Development

Possible future improvements include:

- Improve hand-to-face detection accuracy.
- Collect more representative sneeze and background-audio recordings.
- Improve audio classification reliability.
- Develop a dedicated ROS 2 subscriber for the robot behavior subsystem.
- Test the complete perception-to-behavior trigger.
- Integrate the trigger with the simulated tissue-delivery robot.
