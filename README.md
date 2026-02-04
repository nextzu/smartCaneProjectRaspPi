# Visual Obstacle Detection System for Blind Navigation Project AKA Smart Cane 
# Zeynep Uzun 
This project implements a real-time obstacle detection and navigation assistance system designed to support visually impaired users during indoor and outdoor navigation. The system combines computer vision, deep learning-based object detection and edge-server computing to provide immediate audio and tactile feedback about obstacles in the user's path.

A Raspberry Pi-based client captures camera frames and transmits them to a locally running server that performs object detection using YOLOv8. The server analyzes obstacle proximity and direction and returns navigation commands that are translated into audio announcements and LED feedback. The reason of the local execution on server side is because of the speak module.

---

## System Overview

**Client-Server Architecture**

Raspberry Pi (Client)
  - Captures camera frames
  - Compresses images and sends them to the server
  - Provides LED-based haptic feedback

Windows Server
  - Runs YOLOv8 object detection
  - Computes obstacle proximity and direction
  - Generates audio feedback using text-to-speech
  - Streams processed frames for debugging

---

## Requirements

### Raspberry Pi (Client)

Python packages:
```
requests
numpy
pillow
```

System packages (recommended):
```
sudo apt update
sudo apt install -y libjpeg-dev zlib1g-dev
```

---

### Server (Windows)

Python packages:
```
flask
numpy
opencv-python
ultralytics
```

---

## Running the System

### Raspberry Pi (Client)
One important note is that the IP should match the server machine IP. We advise to run ipconfig on the server machine first and then paste the IP seen there to the SERVER_IP. 
```
python pi_client13.py
```

---

### Server (Windows - PowerShell)
We ran this on local because the speak module functions on local not Docker.

```
$env:SPEAK="1"
$env:LEFT_THRESH="0.33"
$env:RIGHT_THRESH="0.67"
python server_code13.py
```

---

## Live Stream 

Go to http://127.0.0.1:8000/stream to watch the frames incoming.

## Environment Variables

SPEAK -> Enable (1) or disable (0) text-to-speech 

LEFT_THRESH -> Horizontal boundary for left obstacle zone 

RIGHT_THRESH -> Horizontal boundary for right obstacle zone 

---

## Features

- Real-time obstacle detection using YOLOv8
- Custom proximity estimation without depth sensors
- Direction-aware alerts (left, front, right)
- Audio feedback with anti-spam logic
- LED-based haptic feedback
- MJPEG video stream for live debugging
- Configurable thresholds via environment variables

---

