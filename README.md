# Number Gesture Recognition Robot (0-5) 🤖✋

A lightweight, coordinate-based gesture recognition system designed to control a Raspberry Pi robot using hand signs. This project moves away from traditional CNNs, instead utilizing **Google MediaPipe** for skeletal extraction and a custom **Multi-Layer Perceptron (MLP)** for classification, achieving **98.83% accuracy**.

---

## 📝 Abstract
Human-Robot Interaction (HRI) increasingly demands intuitive interfaces. Traditional vision-based systems often rely on processing full raw images, which leads to significant noise and high computational costs.

This project adopts a **Coordinate-based AI** approach. Instead of processing entire image frames, we use MediaPipe to extract the spatial coordinates of hand joints (21 landmarks), reducing the input from thousands of pixels to a compact vector of coordinates. The model was trained in TensorFlow/Keras and converted to **ONNX** for efficient deployment on a robot.

## 🚀 Key Features
* **Coordinate-Based Inference:** Extracts 21 3D hand landmarks (x, y, z) to create a feature vector of size 63.
* **High Accuracy:** Achieved **98.83% Test Accuracy** on a custom dataset of 3,000 samples.
* **Translation Invariance:** Implements relative coordinate normalization (setting wrist to 0,0,0) to ensure the model learns hand shape, not position.
* **Optimized Deployment:** Converted to ONNX Runtime, achieving a **20% reduction** in inference latency.

## 🛠️ System Architecture

**The Vision Pipeline:**
`Webcam` → `MediaPipe (Landmarks)` → `Preprocessing (Normalization)` → `ONNX Model` → `Robot Command`.



### Neural Network Structure (MLP)
We utilized a Feed-Forward Network optimized for the classification of 6 distinct gesture classes.

| Layer Type | Nodes / Rate | Activation | Description |
| :--- | :--- | :--- | :--- |
| **Input** | 63 | - | Flattened vector (21 x 3) |
| **Dense** | 128 | ReLU | First hidden layer |
| **Dropout** | 0.3 | - | Prevents overfitting |
| **Dense** | 64 | ReLU | Second hidden layer |
| **Dropout** | 0.3 | - | Prevents overfitting |
| **Output** | 6 | Softmax | Class probabilities (0-5) |






## 🕹️ Controls & Mapping
The system maps recognized Class IDs to velocity commands sent via Wi-Fi Sockets.

| Class ID | Gesture | Robot Action |
| :---: | :--- | :--- |
| **0** | Open Palm | **Idle / Stop** |
| **1** | Index Finger Up | **Move Forward (Slow)** |
| **2** | Index + Middle | **Move Forward (Medium)** |
| **3** | Three Fingers | **Move Forward (Fast)** |
| **4** | Four Fingers (Thumb Tucked) | **Turn Left** |
| **5** | High Five / Open Hand | **Turn Right** |

## 📊 Performance & Results
* **Latency:** Local inference runs at approximately **35ms** (~28 FPS). However, due to network constraints, effective robot control averaged **3-5 FPS**.
* **Confusion Matrix Analysis:**
    * **Class 0 (Stop):** 100% Precision/Recall.
    * **Class 4 vs 5:** Minor misclassification occurred between "Turn Left" and "Turn Right" due to geometric similarities in the thumb position.

## ⚙️ Installation & Usage

### 1. Prerequisites
* Python 3.8+
* TensorFlow 2.x
* MediaPipe
* ONNX Runtime
* OpenCV

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/ponitaponlork/Number-Gesture-Recognition-Robot.git
cd gesture-robot

# Install dependencies
pip install tensorflow mediapipe opencv-python onnxruntime tf2onnx
