

---

# 🎧 Signal and Image Processing Toolkit

A Python-based interactive system that demonstrates fundamental concepts of **Digital Signal Processing (DSP)** through audio and image processing tasks. This project enables real-time audio recording, signal combination, playback rate manipulation, and analysis of systems for linearity, time invariance, and frequency-domain characteristics.

---

## 📌 Features

### 🎤 Audio Signal Processing

* **Live Audio Recording** using `pyaudio`
* **Choose Existing Audio Files** from the system
* **Combine Two Audio Signals** after resizing
* **Playback Rate Modification** (time-scaling)
* **Frequency Analysis** using custom DTFT implementation
* **System Property Checks**:

  * Linearity (Superposition & Homogeneity)
  * Time Invariance

### 📸 Image Processing

* **Live Webcam Feed** using `OpenCV`
* **Capture Image on Key Press ('s')**
* **Exit Webcam on Key Press ('q')**

---

## 📁 Directory Structure

```
project/
│
├── your_script.py              # Main Python file
├── captured_image.jpg          # Saved webcam image (on capture)
├── recorded_audioXYZ.wav       # Recorded audio samples
├── combined.wav                # Combined audio signal
├── stretched.wav               # Time-scaled audio
└── README.md                   # This file
```

---

## 🧰 Requirements

Install the following Python packages before running the script:

```bash
pip install pyaudio wave opencv-python numpy matplotlib soundfile
```

Or use a `requirements.txt` file:

```txt
pyaudio
wave
opencv-python
numpy
matplotlib
soundfile
```

---

## 🚀 How to Run

1. Run the Python script:

   ```bash
   python your_script.py
   ```

2. Follow the on-screen prompts to:

   * Record or select audio
   * Perform audio operations (combine, playback scaling, DTFT, etc.)
   * Use the webcam for image capture

---

## 🧪 DSP Concepts Covered

| Operation                 | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| **Addition of Signals**   | Combines two audio signals after resizing                    |
| **Time Scaling**          | Modifies playback rate (speed up or slow down)               |
| **Linearity Check**       | Tests superposition and homogeneity                          |
| **Time Invariance Check** | Tests if shifting input results in shifted output            |
| **DTFT Analysis**         | Custom implementation of DTFT with magnitude and phase plots |

---

## 📷 Webcam Functionality

* Live feed using OpenCV
* Press `'s'` to save the frame as `captured_image.jpg`
* Press `'q'` to exit

---

## 👨‍💻 Contributors

| Name                   | Roll Number      |
| ---------------------- | ---------------- |
| Christin Prakash       | CB.EN.U4ECE24208 |
| Dinesh Kuthalanathan T | CB.EN.U4ECE24211 |
| Gokul S                | CB.EN.U4ECE24212 |
| Hariprasad G           | CB.EN.U4ECE24213 |

---

## 📌 Notes

* All paths are hardcoded for `/home/hariprasad/`; adjust if running on a different system.
* Audio files should be `.wav` format for compatibility.

---

## 📜 License

This project is for academic and educational purposes only.

---
