# 🚗 Vehicle Detection System

A real-time traffic monitoring, vehicle detection and counting system.

🔗 **Live Demo:** https://vehicle-detector-15st.onrender.com

---

## Features

- 🚗 Detects and classifies vehicles — Car, Motorcycle, Bus, Truck
- 📸 Supports Image, Video, and Webcam input
- 📊 Vehicle count per type
- ⚡ Speed estimation
- 🔲 Adjustable line crossing detection

---

## Tech Stack

- **YOLOv8** — Object detection
- **OpenCV** — Image & video processing
- **Streamlit** — Web interface
- **Python** — Core language

---

## Project Structure
```
vehicle_detection/
├── detector.py      # Detection logic (YOLOv8, OpenCV)
├── app.py           # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Run Locally
```bash
# Clone repo
git clone https://github.com/ahmmodshihab/vehicle-detector.git
cd vehicle-detector

# Virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## How It Works
```
Input (Image/Video/Webcam)
        ↓
YOLOv8 detects vehicles
        ↓
OpenCV draws bounding boxes
        ↓
Count & speed calculated
        ↓
Streamlit displays result
```

---

## Screenshots
![Vehicle Detection](<Screenshot 2026-03-26 120937.png>)

---

*Built by Shihab | Computer Vision Engineer*