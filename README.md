
# 📊 Crowd Management using a Digital Twin Approach

This project presents a digital twin–based system for real-time crowd monitoring and emergency prediction using deep learning. It leverages a **CSRNet-based density estimation model** to process video feeds in highly dense religious gatherings like the **Mahakumbh Mela**.

---

## 🧠 Project Objectives

- Monitor **live crowd density** from video feeds.
- Estimate **people count and congestion zones** using CSRNet.
- Trigger **real-time emergency alerts** (visual + beep + WhatsApp/SMS) on crowd spike detection.
- Validate and visualize predictions with evaluation tools.
- Provide a **digital twin**–like representation using 2D density heatmaps.

---

## 🗂️ Repository Contents

| File/Folder           | Purpose |
|-----------------------|---------|
| `main.py`             | Real-time inference on video + emergency spike detection and alerts |
| `eval.py`             | Evaluation of model performance on test data (MAE, RMSE, pseudo accuracy) |
| `visualize_results.py`| Generates prediction vs ground truth visualizations |
| `train.py`            | Script to train CSRNet on custom or benchmark dataset |
| `model.py`            | CSRNet model definition (based on VGG16 frontend + dilated backend) |
| `utils.py`            | Model saving/checkpoint utilities |
| `notifier.py`         | Triggers beep + SMS/WhatsApp alerts (requires credentials) |
| `requirements.txt`    | All Python dependencies for the project |

---

## 📷 Pretrained Model .pth file and sample Input/Output Files

📁 Find the sample input videos and result visualizations along with the model .pth file here:

➡️ **[Sample Input/Output Files on Google Drive](https://drive.google.com/drive/folders/1M-Z79b7fjOM091HWcF5rcXDAj1osWAXv?usp=sharing)**

---

## 📁 Dataset

The model is evaluated on the **ShanghaiTech Part A dataset**, a standard benchmark for crowd counting.

🔗 **[Download Train/Test Data (Google Drive)](https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view)**

Once downloaded:
- Use JSON annotations for `train.py`
- Use image and `.mat` files for `eval.py` and `visualize_results.py`

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
```bash
git clone https://github.com/Anjaneya-Pattanaik/Crowd-Management-using-a-Digital-Twin-Approach.git
cd crowd-digital-twin #root folder path
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download Pretrained Model and Sample Inputs**

To run the project directly without training:
- Download the pretrained model (PartAmodel_best.pth)
- Download sample video and evaluation test data

Place the downloaded .pth file in your project root directory as PartAmodel_best.pth.

---

## 📁 File Path Instructions

### 📝 Important:
Throughout the code files (e.g., main.py, eval.py, train.py, visualize_results.py), there are placeholders like:

"path to this file"
"path/to/train.json"
"video.mp4"
"test_data/images"

### 🔧 Before running the project, replace these with the actual paths to your:

✅ Input video file (e.g., video.mp4)

✅ Ground truth images and .mat files

✅ Pretrained model file (PartAmodel_best.pth)

✅ Training and validation annotation .json files

---

## 🚀 Running the Project

### 🔎 Real-Time Inference with Emergency Spike Detection
```bash
python main.py
```

- Loads the input video
- Displays crowd heatmap with count
- Triggers alerts (beep + optional SMS/WhatsApp) if crowd count spike detected

### 🧪 Evaluate on Test Dataset
```bash
python eval.py
```

- Computes MAE, RMSE, and pseudo accuracy using `.mat` ground truth files

### 📊 Visualize Sample Results
```bash
python visualize_results.py
```

- Generates predicted vs. actual density maps
- Saves as `visual_result_1.png`, `visual_result_2.png`, ...

### 🧠 Training CSRNet

1. Make sure you have:
- part_A_train.json and part_A_val.json
- Images and their corresponding .h5 density maps
  
2. Run the following command to begin training:
```bash
python train.py part_A_train.json part_A_val.json --pre None 0 run1
```

- Trains CSRNet using ShanghaiTech or custom dataset
- Saves checkpoints and best model under task `run1`

---

## 📌 Features

- ✅ CSRNet for crowd counting
- ✅ Spike-based emergency detection algorithm
- ✅ Realtime heatmap visualization
- ✅ Evaluation metrics: MAE, RMSE, pseudo accuracy
- ✅ Training script and model loader
- ✅ WhatsApp and SMS alert integration via Twilio API (requires setup)

---

## 📊 Evaluation Summary

| Metric          | Value (Example) |
|------------------|----------------|
| MAE (Test Set)   | 85.54          |
| RMSE (Test Set)  | 142.60         |
| Pseudo Accuracy  | ~63% (within ±10% tolerance) |

---

_**Note: This project was developed and submitted as part of the B.Tech Minor Project under the School of Computer Sciences, Odisha University of Technology and Research (2025).**_
