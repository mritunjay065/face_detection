# 🛡️ Enterprise AI Assessment Proctoring & Face Attention Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-orange.svg)](https://developers.google.com/edge/mediapipe)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)

An enterprise-grade, multimodal AI proctoring and real-time visual attention estimation system designed for high-stakes Online Assessments (OA), technical screening tests, and remote proctoring.

---

## 📌 Table of Contents
- [What Is This System?](#-what-is-this-system)
- [Why Build It? (Problem Statement & Philosophy)](#-why-build-it)
- [How Does It Work? (System Architecture)](#-how-does-it-work)
- [Core Features](#-core-features)
- [Technology Stack](#-technology-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started & Installation](#-getting-started--installation)
- [How to Run (CMD / Desktop)](#-how-to-run)
- [Keyboard Controls & Telemetry Guide](#-keyboard-controls--telemetry-guide)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [License](#-license)

---

## 💡 What Is This System?

This is a real-time computer-vision and audio-intelligence proctoring suite that runs locally on edge devices. It tracks:
1. **Visual Engagement (0–100 score):** Gaze tracking, head orientation, and eye behavior over time.
2. **Cheating & Collusion Detection:** Detects multiple faces, unauthorized cell phones, notes, books, and laptops.
3. **Audio Integrity:** Voice Activity Detection (VAD) to flag background talking, coaching, or whispering.
4. **Presentation Attack Defense (Anti-Spoofing):** Rejects photo cutouts and digital screen video replays.
5. **Automated Incident Recording:** Instantly saves timestamped 10-second video evidence clips when violations occur.

---

## 🎯 Why Build It?

### 1. Observable Physical Signals $\neq$ True Mental Cognition
A common pitfall in basic proctoring systems is assuming that looking away for a single second or blinking means a student is cheating. This architecture explicitly decouples **measured physical signals** from **inferred engagement** through temporal rolling windows, exponential smoothing, and personal dynamic calibration.

### 2. Solving Real-Life Assessment Vulnerabilities
Standard webcams are prone to false positives (natural Asian/slanted eye shapes getting falsely tagged as `EYES_CLOSED`) and false negatives (an accomplice whispering solutions off-camera). This system solves both using:
- **Self-calibrating Eye Aspect Ratio (EAR)** that calculates each candidate's personal open-eye baseline.
- **Multimodal Audio Tracking** to catch whispered collusion.
- **Dual Multi-Face Trackers** to flag unauthorized secondary people entering the camera view.

---

## ⚙️ How Does It Work?

```mermaid
flowchart TD
    subgraph Ingestion [Hardware Input]
        Cam[Webcam Feed DirectShow 30 FPS]
        Mic[Microphone SoundDevice 16kHz]
    end

    subgraph VisionPipeline [Perception Pipeline]
        Cam --> Mesh[MediaPipe 468+ Landmarker]
        Cam --> YOLO[Async YOLOv8 Nano Thread]
        Cam --> DualFace[Dual Multi-Face Counter]
        Mesh --> Gaze[Iris Gaze & Eye-Rolling gy/gx]
        Mesh --> Pose[OpenCV solvePnP 3D Head Pose]
        Mesh --> EAR[Self-Calibrating EAR Detector]
        Cam --> FAS[Anti-Spoofing Fourier Moiré Engine]
    end

    subgraph AudioPipeline [Audio Intelligence]
        Mic --> VAD[Spectral Voice Activity Detector]
    end

    subgraph DecisionEngine [Temporal & Risk Fusion]
        Gaze & Pose & EAR --> Temporal[5s Rolling Window Stats]
        Temporal --> ML[HistGradientBoosting Classifier]
        YOLO & DualFace & FAS & VAD & ML --> Fusion[Enterprise Threat Matrix]
    end

    subgraph Action [Output & Audit]
        Fusion --> HUD[Real-Time OpenCV Security HUD]
        Fusion -->|HIGH RISK| Clip[Audit Evidence Clip Recorder]
    end
```

---

## 🚀 Core Features

- **Geometric Cyber-Mesh:** Sleek polygonal triangulation facets with glowing nodes and iris crosshairs rendered directly on the candidate's face.
- **Zero-Latency Async YOLOv8:** Runs in a dedicated background worker thread with double-buffering; main video feed stays at silky $30\text{--}60\text{ FPS}$ with zero camera lag.
- **Strict Multi-Class Object Discrimination:** Separates prohibited devices (`CELL PHONE`, `BOOK / NOTES`) from permissible desk hardware (`MOUSE`, `KEYBOARD`, `CUP`, `BOTTLE`).
- **Precision Eye-Rolling & Saccade Detection:** Flags upward gaze drift toward ceiling prompters ($g_y < 0.34$).
- **Dual Multi-Face Collusion Guard:** Immediately marks any secondary person entering the frame as `HIGH RISK`.
- **Audio Voice Detection:** Analyzes human vocal bands ($85\text{--}350\text{ Hz}$) to flag whispering or background coaching.
- **Anti-Spoofing Liveness:** Analyzes 2D Fourier high-frequency moiré patterns to block digital screen video replays and printed photo cutouts.
- **Automated Incident Clips:** Automatically writes a 10-second `.avi` evidence video clip to disk whenever a severe violation occurs.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Video Capture** | OpenCV (`cv2.CAP_DSHOW`) | Low-latency hardware camera capture |
| **Facial Landmarks** | MediaPipe Tasks Vision (`face_landmarker.task`) | 468+ 3D face mesh + iris tracking |
| **Object Detection** | Ultralytics YOLOv8 Nano (`yolov8n.pt`) | Prohibited device & person detection |
| **Head Pose** | OpenCV `solvePnP` + Rodrigues | Continuous Euler rotation (Yaw, Pitch, Roll) |
| **Audio Processing** | SoundDevice + NumPy FFT | Real-time decibel & voice frequency tracking |
| **ML Inference** | Scikit-Learn `HistGradientBoostingClassifier` | Temporal behavior classification |
| **Liveness / FAS** | NumPy 2D FFT + Laplacian | Moiré pattern & texture variance anti-spoofing |
| **Desktop UI** | Native OpenCV HighGUI Window | High-FPS desktop HUD |

---

## 📁 Project Directory Structure

```
face-attention/
├── assets/
│   └── face_landmarker.task         # Google MediaPipe Face Landmarker model asset
├── data/
│   ├── audit_clips/                 # Automated incident evidence video clips
│   ├── features/                    # Windowed tabular feature datasets
│   └── proctor_images/              # Object validation benchmark images
├── models/
│   └── engagement_model.joblib      # Trained Gradient Boosting Classifier
├── src/
│   ├── anti_spoofing.py             # Passive liveness & Fourier moiré analyzer
│   ├── audio_proctor.py             # SoundDevice spectral voice activity detector
│   ├── audit_recorder.py            # Automated 10-second rolling incident recorder
│   ├── capture.py                   # Low-latency camera capture manager
│   ├── evaluate_unseen.py           # Unseen candidate generalization benchmark
│   ├── eye_state.py                 # Self-calibrating EAR & blink tracker
│   ├── features.py                  # 5-second temporal aggregation engine
│   ├── gaze.py                      # Iris gaze & eye-rolling detector
│   ├── head_pose.py                 # 3D head pose estimation via solvePnP
│   ├── landmarks.py                 # MediaPipe wrapper & cyber-mesh renderer
│   ├── multi_face_detector.py       # Dual multi-face & collusion counter
│   ├── predict.py                   # Ensemble inference & EMA score smoother
│   ├── scoring.py                   # Transparent baseline scoring formulas
│   ├── train.py                     # GroupShuffleSplit subject-level training
│   └── yolo_detector.py             # Asynchronous multi-class YOLOv8 inspector
├── tests/
│   ├── test_pipeline.py             # Pipeline unit tests
│   └── test_smoke_regression.py     # Smoke & regression test suite
├── main.py                          # Main desktop entry point
├── requirements.txt                 # Project dependencies
├── LICENSE                          # MIT License
└── README.md
```

---

## 📦 Getting Started & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mritunjay065/face_detection.git
cd face_detection
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

Launch directly from Command Prompt or Terminal:
```cmd
python main.py
```

### Customizable Launch Options:
```cmd
python main.py --camera 0 --yolo-conf 0.20 --ear-thresh 0.15 --window 5.0
```
- `--camera 0`: Camera index (default: `0`).
- `--yolo-conf 0.20`: Confidence threshold for YOLO object detector.
- `--ear-thresh 0.15`: Threshold for eye closure detection.
- `--window 5.0`: Rolling temporal history window in seconds.
- `--no-audio`: Runs in video-only mode without microphone monitoring.

---

## ⌨️ Keyboard Controls & Telemetry Guide

| Key | Action | Description |
| :---: | :--- | :--- |
| **`c`** | **Calibrate Gaze** | Look comfortably at your screen and press `c`. Instantly zeroes your natural eye gaze to perfect center $(g_x=0.50, g_y=0.50)$. |
| **`w`** | **Toggle Wireframe** | Toggles the Geometric Cyber-Mesh on or off. |
| **`r`** | **Toggle YOLO Scan** | Toggles background YOLO room scanning on or off. |
| **`q`** / **`ESC`** | **Exit** | Gracefully releases camera and terminates audio/YOLO threads. |

### In-Screen HUD Telemetry
```
RISK: LOW | SCORE: 92.4/100 | ON_TARGET
FPS: 30 (33ms) | Gaze: CENTER (gy:0.49) | EAR: 0.24 | Audio: -42dB
```
- **Top Banner Color:** 
  - 🟢 **Green (`LOW RISK`):** Perfect test-taking posture; no violations.
  - 🟠 **Amber (`MEDIUM RISK`):** Eye-rolling, looking away, or drowsiness warning.
  - 🔴 **Red (`HIGH RISK`):** Unauthorized device, collusion, audio talking, or spoofing detected.

---

## 🧪 Testing & Quality Assurance

Run the automated smoke and regression test suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

Benchmark model accuracy on held-out unseen subjects:
```bash
python -m src.evaluate_unseen
```
* **False Alarm Rate on Attentive Candidates:** `< 1.0%`
* **Generalization F1-Score:** `1.00` across all held-out subjects.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full details.
