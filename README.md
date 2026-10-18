# Real-Time Lane Perception and Collision Avoidance for Autonomous Vehicles

> [!CAUTION]
> ### ⚠️ PROPRIETARY NOTICE & STRICT COPYRIGHT (ALL RIGHTS RESERVED)
> **Copyright © 2026 KARTHIK K S. All Rights Reserved.**  
> **Strictly NO copying, cloning, redistribution, modification, or reuse** of this code, algorithms, or architectures is permitted for academic submissions, projects, or commercial use without prior express written permission from the author.  
> *This repository is made publicly viewable strictly for portfolio verification and demonstration purposes only.*  
> For permission requests, contact: **karthikkornengattil@gmail.com**

---

A lightweight, multi-task deep learning perception and control system designed for autonomous vehicles operating under real-time embedded constraints.

---

## 🚀 Overview

This repository implements a concurrent tripartite vision perception and control pipeline:
1. **Dynamic Lane Perception & Segmentation**: A lightweight U-Net with Depthwise Separable Convolutions and a 10-step Bidirectional LSTM (BiLSTM) bottleneck for continuous temporal lane tracking under severe shadows and occlusions.
2. **Multi-Class Obstacle Detection**: A fast Depthwise CNN classifier operating on region proposals to detect and categorize surrounding traffic obstacles into 6 classes (*Car, Pedestrian, Cyclist, Bus, Van, Truck*).
3. **End-to-End Steering Angle Prediction**: A 5-convolution regression neural network based on the NVIDIA architecture for smooth, closed-loop lateral steering control.

---

## 📊 System Architecture & Performance

```
                           CAMERA RGB STREAM (30 FPS)
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
  [Lane Perception]          [Obstacle Detection]     [Steering Prediction]
  Lightweight U-Net +        6-Class Depthwise CNN    NVIDIA Regression CNN
  10-Step BiLSTM Bottleneck  Spatial NMS (IoU=0.45)   Flip-Negate Augmentation
  ─────────────────────────  ─────────────────────    ───────────────────────
  • Acc: 96.52%              • Acc: 94.20%            • MSE: 0.012 rad²
  • Recall: 90.73%           • Latency: 14.5 ms       • MAE: 4.47°
  • Latency: 29.3 ms         • Output: Bounding Boxes • Latency: 18.2 ms
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      ▼
                        [DECISION & FUSION ENGINE]
                  Synchronized Multi-Threaded Execution
                  Real-Time Telemetry & Visual Overlay
                                      │
                                      ▼
                           [SERIAL ARBITRATION]
                  Packet: "$STR,-16.3,SPD,45,BRAKE,0#"
                  Baud: 115200 | Latency: 1.8 ms
                                      │
                                      ▼
                        [ARDUINO ACTUATION BRIDGE]
                     L293D / L298N Motor & Servo Control
                   + Dedicated Sonar Interrupt (<20 cm Stop)
```

### Key Performance Benchmarks
* **Lane Detection Accuracy**: 96.52% pixel accuracy, 90.73% safety recall on the TuSimple benchmark.
* **Model Footprint**: Reduced by 87% (only **1.12M parameters**, ~4.3 MB RAM).
* **Obstacle Classification**: 94.20% weighted accuracy across 6 KITTI traffic classes in **14.5 ms**.
* **Steering Regression**: Mean Absolute Error of **4.47°**, Pearson correlation $r = 0.964$ in **18.2 ms**.
* **Embedded Throughput**: Sustains **34.1 FPS** on the NVIDIA Jetson Orin Nano.

---

## 📁 Repository Structure

```
.
├── LICENSE                                   # Strict All-Rights-Reserved proprietary license
├── README.md                                 # Project documentation & copyright notice
├── requirements.txt                          # Python package dependencies
├── complete_autonomous_system.py             # Master multi-threaded pipeline combining all 3 modules
├── lane_detection_lstm_balanced_20260621.py  # Lightweight U-Net + BiLSTM temporal lane model
├── lane_detection.py                         # Spatial lane detection baseline
├── lane_detection_lstm.py                    # Standard LSTM temporal lane model
├── lane_steering_combined.py                 # Integrated lane perception + steering regression
├── kitti_cnn_final.py                        # 6-class KITTI obstacle detection model
└── steering_prediction.py                    # NVIDIA end-to-end steering angle regressor
```

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* NVIDIA GPU with CUDA support (recommended for real-time inference)
* Webcam or recorded highway driving video stream

### 1. Clone the Repository
```bash
git clone https://github.com/karthikks2004/Autonomous-Vehicle-Lane-Perception-CNN-BiLSTM.git
cd Autonomous-Vehicle-Lane-Perception-CNN-BiLSTM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚦 How to Run

### Run the Master Autonomous System:
```bash
python complete_autonomous_system.py
```

### Run Individual Modules:
* **Lane Detection (Lightweight U-Net + BiLSTM):**
  ```bash
  python lane_detection_lstm_balanced_20260621.py
  ```
* **Obstacle Detection (6-Class KITTI CNN):**
  ```bash
  python kitti_cnn_final.py
  ```
* **Steering Angle Prediction (NVIDIA CNN):**
  ```bash
  python steering_prediction.py
  ```

---

## 🔒 Safety & Failsafe Design
The system employs a defense-in-depth architecture:
* High-level visual perception and multi-threaded arbitration run on the primary compute unit (Jetson Orin Nano).
* Serial packets formatted as `$STR,<angle>,SPD,<speed>,BRAKE,<flag>#` are transmitted via UART to an Arduino microcontroller.
* An independent ultrasonic sonar sensor operates on a 10 ms hardware interrupt on the microcontroller to force an immediate emergency motor shutdown whenever an obstacle is detected within 20 cm, protecting the vehicle even during operating system or camera frame drops.

---

## ⚖️ Legal & Copyright
This repository and its codebase are strictly protected under international copyright law. Unauthorized copying, distribution, or submission of this project without explicit written authorization is strictly prohibited. See the [LICENSE](LICENSE) file for complete legal terms.

* **Author:** KARTHIK K S  
* **Affiliation:** Department of Electronics and Communication Engineering, Amrita School of Engineering, Coimbatore, India  
* **Contact:** `karthikkornengattil@gmail.com`
