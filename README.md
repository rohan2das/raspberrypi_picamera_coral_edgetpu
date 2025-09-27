# Real-time Object Detection with Raspberry Pi and Coral Edge TPU

This project demonstrates a real-time object detection system that streams video from a camera on one device to a Raspberry Pi connected to a Coral Edge TPU for accelerated inference.

The system is split into two main components:
-   **`camera_sender`**: Captures video frames and streams them over a network socket.
-   **`coral_receiver`**: Receives the frames, performs object detection using the Coral Edge TPU, and displays the results.

---

## Directory Structure
```
.
├── camera_sender/
│   ├── send_camera_frames.py
│   └── requirements_picam.txt
├── coral_receiver/
│   ├── receive_and_infer.py
│   ├── requirements_coral.txt
│   ├── 240_yolov8n_full_integer_quant_edgetpu.tflite
│   └── coco.txt
└── socket_interface/
    └── ... (shared socket connection files)
```

---

## Prerequisites

-   Raspberry Pi (or similar Single Board Computer)
-   Google Coral USB Accelerator
-   A camera module compatible with your sender device (e.g., Raspberry Pi Camera Module)

---

## Setup Instructions

This project requires two separate environments: one for sending the camera feed and another for receiving and processing it on the Coral device.

### 1. `camera_sender` Setup

1.  **Navigate to the directory:**
    ```bash
    cd camera_sender
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv_picam
    source venv_picam/bin/activate
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements_picam.txt
    ```

---

### 2. `coral_receiver` Setup (on Raspberry Pi)

The Coral Edge TPU runtime library requires a specific Python version (**3.6 - 3.9**). If your Raspberry Pi is running a newer Python version (e.g., 3.11), you must first install a compatible version. The following steps use **`pyenv`** to install and manage Python 3.9 without altering the system's default Python installation.

#### Step A: Install `pyenv` and Dependencies

1.  **Update system packages:**
    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    ```
2.  **Navigate into the `coral_receiver` directory:**
    ```bash
    cd coral_receiver
    ```
3.  **Install `pyenv`:**
    ```bash
    curl [https://pyenv.run](https://pyenv.run) | bash
    ```
4.  **Configure your shell environment for `pyenv`**. This adds `pyenv` to your path for the current session and for future logins.
    ```bash
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    exec "$SHELL"
    ```
5.  **Install system dependencies** required to build Python from source:
    ```bash
    sudo apt-get install --yes libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libgdbm-dev lzma lzma-dev tcl-dev libxml2-dev libxmlsec1-dev libffi-dev \
    liblzma-dev wget curl make build-essential openssl
    ```

#### Step B: Install Python 3.9 and Create Environment

1.  **Install Python 3.9.12 using `pyenv`**. This command will download and build Python, which can take a significant amount of time.
    ```bash
    pyenv install 3.9.12
    ```

3.  **Set the local Python version for this directory**. `pyenv` will now automatically use Python 3.9.12 whenever you are in this folder.
    ```bash
    pyenv local 3.9.12
    ```
4.  **Verify that the correct Python version is active:**
    ```bash
    python --version
    # Expected output: Python 3.9.12
    ```
5.  **Create and activate a new virtual environment** using Python 3.9:
    ```bash
    python -m venv venv_picam
    source venv_picam/bin/activate
    ```
6.  **Install the required Python packages** for the receiver, including the Coral libraries:
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run


1.  **On the camera folder**, start the sender script:
    ```bash
    cd camera_sender
    source venv/bin/activate
    python send_camera_frames.py
    ```


2.  **On the coral folder**, start the receiver script:
    ```bash
    cd coral_receiver
    source venv_picam/bin/activate
    python receive_and_infer.py
    ```

An OpenCV window should now appear on the receiver's display, showing the live video stream with object detection bounding boxes.
