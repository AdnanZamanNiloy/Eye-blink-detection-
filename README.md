
# 👁️ Eye-Blink Detection

A real-time **eye-blink detection system** built using computer vision. This project detects blinks from a live webcam feed using facial landmarks and can be used for applications like **drowsiness detection**, **liveness detection**, and **attention monitoring**.

---

## 🚀 Features

✔️ Real-time blink detection using webcam
✔ Detects eyes using facial landmarks
✔ Shows blink count on screen
✔ Lightweight and fast
✔ Built with OpenCV and Dlib
✔ Can be extended for driver drowsiness systems or liveness checks

---

## 🧠 How It Works

1. Capture video from webcam.
2. Detect the face in each frame using a face detector.
3. Find facial landmarks around the eyes.
4. Measure the eye aspect ratio (EAR) to determine open/closed status.
5. If EAR stays below a threshold for a blink, count it.

---

## 📦 Tech Stack

| Component        | Technology                              |
| ---------------- | --------------------------------------- |
| Language         | Python                                  |
| Computer Vision  | OpenCV                                  |
| Facial Landmarks | Dlib                                    |
| Model Files      | `shape_predictor_68_face_landmarks.dat` |

---

## 🛠️ Prerequisites

Before running, make sure you have:

✔ Python 3.8+
✔ Webcam connected

---

## 📥 Installation

1. **Clone the repository**

```bash
git clone https://github.com/AdnanZamanNiloy/Eye-blink-detection.git
cd Eye-blink-detection
```

2. **Create and activate a Python virtual environment**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download required model file**

Download `shape_predictor_68_face_landmarks.dat` and place it inside the project folder.

You can download it from:
➡️ [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Extract the file after downloading.

---

## Run the Project

```bash
python blink_detection.py
```

📍 The webcam will open in a new window, and blinks will be counted live.

---

## 🧪 Usage

✔ Keep your face in the webcam view
✔ Blink naturally
✔ The program will show the blink count on screen

---

## 📸 Screenshot (Optional)

*You can add a screenshot of the detection working here.*

---

## 💡 Potential Use Cases

✔ Driver drowsiness detection
✔ Security systems with liveness checks
✔ Attention monitoring for remote exams
✔ Interactive UI controls (e.g., blink to trigger something)

---

## 🧩 Code Structure

```
Eye-blink-detection/
├── blink_detection.py        # Main detection script
├── requirements.txt          # Python dependencies
├── shape_predictor_68_face_landmarks.dat  # Dlib model file (not included)
└── LICENSE
```

---

## 📚 Dependencies

You can install manually with:

```bash
pip install opencv-python
pip install dlib
pip install imutils
```

Or just use:

```bash
pip install -r requirements.txt
```
