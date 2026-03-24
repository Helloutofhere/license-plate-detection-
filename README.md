# 🚗 Automatic Number Plate Recognition (ANPR) System

## 📌 Overview
Automatic Number Plate Recognition (ANPR) is a computer vision system designed to detect and recognize vehicle license plates in real time.

This project uses **YOLOv8 for detection** and **OCR (EasyOCR/PaddleOCR)** for extracting alphanumeric characters from Indian number plates.

---

## 🎯 Objectives
- Detect number plates from images/videos  
- Extract text using OCR  
- Handle real-world challenges like blur, lighting, and variations  

---

## 🧠 Methodology

### 🔹 Step 1: Data Collection & Annotation
- Collected dataset of Indian vehicles  
- Annotated plates in YOLO format  

### 🔹 Step 2: Preprocessing
- Grayscale conversion  
- Noise reduction  
- Contrast enhancement  

### 🔹 Step 3: Detection (YOLOv8)
- Real-time plate detection  
- Bounding box extraction  

### 🔹 Step 4: OCR (Text Recognition)
- EasyOCR / PaddleOCR used  
- Extracts alphanumeric text  

### 🔹 Step 5: Output
- Displays plate number  
- Stores results in database/CSV  

---

## 🛠️ Tech Stack
- Python  
- OpenCV  
- YOLOv8 (Ultralytics)  
- EasyOCR / PaddleOCR  
- NumPy, Pandas  

---

## 📊 Results
- High accuracy detection  
- Real-time performance achieved  
- Robust under varying conditions  

---

## ⚠️ Challenges
- Plate variation (Indian formats)  
- Lighting & glare issues  
- Motion blur  
- OCR misclassification  

---

## 📷 Screenshots

![vehDetection](1.png)
![Plate](2.png)
![OCR](3.png)

---

## 👩‍💻 My Contribution
- Worked on detection pipeline using YOLOv8  
- Implemented OCR integration  
- Improved preprocessing for accuracy  

---

## ⚠️ Note
Due to system constraints, full training code and datasets are not uploaded. This repository showcases project workflow and outputs.

---

## 🚀 Future Scope
- Transformer-based OCR (TrOCR)  
- Real-time traffic integration  
- Smart city deployment  
- Fraud detection  

---

## 📄 Documentation
Detailed project report is included in this repository.
