import platform
import os
import time
import torch
import cv2
import numpy as np
from torchvision import transforms
from model import CSRNet
from notifier import send_sms_alert, send_whatsapp_alert

# Emergency Prediction Initialization
from collections import deque

# ----------- PARAMETERS (tune as needed) -----------
window_size = 2         # Number of frames for moving statistics - 20
spike_sigma = 0          # Spike threshold: mean + 2*std - 2
spike_persistence = 1    # Number of consecutive spikes before alert - 3
alert_cooldown = 60      # Minimum seconds between alerts

count_history = deque(maxlen=window_size)
spike_counter = 0
last_alert_time = 0

# ----------- DEVICE SETUP -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- LOAD MODEL -----------
model = CSRNet().to(device)
checkpoint = torch.load('path to PartAmodel_best.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
model.eval()

# ----------- PREPROCESSING FUNCTION -----------
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((768, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(frame).unsqueeze(0).to(device)
    return tensor

# ----------- HEATMAP FUNCTION -----------
def get_heatmap(density_map):
    heatmap = density_map.squeeze().cpu().numpy()
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

# ----------- BEEP ALERT FUNCTION -----------
def play_beep():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # frequency: 1000 Hz, duration: 500 ms
    else:
        print('\a')  # Unix-based: system bell (may not always work)

# ----------- OPEN VIDEO FILE -----------
cap = cv2.VideoCapture('path to input video file')

# ----------- CREATE WINDOW -----------
cv2.namedWindow('Digital Twin - Crowd Heatmap', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Digital Twin - Crowd Heatmap', 1024, 768)

frameCount = -1

while True:
    frameCount += 1
    if (frameCount % 30) != 0:
        continue
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess_frame(frame)
    with torch.no_grad():
        density_map = model(input_tensor)
    count = int(density_map.sum().item())

    # --- Enhanced Emergency Situation Detection ---
    count_history.append(count)
    spike_alert = False

    if len(count_history) == window_size:
        avg_count = np.mean(count_history)
        std_count = np.std(count_history)
        threshold = avg_count + spike_sigma * std_count

        if count > threshold:
            spike_counter += 1
        else:
            spike_counter = 0  # Reset if not persistent

        # Only trigger if spike persists for 'spike_persistence' frames and cooldown has passed
        if spike_counter >= spike_persistence and (time.time() - last_alert_time > alert_cooldown):
            spike_alert = True
            last_alert_time = time.time()
            spike_counter = 0  # Reset after alert

    # --- Visualization ---
    heatmap = get_heatmap(density_map)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    cv2.rectangle(heatmap, (5, 5), (220, 55), (0, 0, 0), -1)
    cv2.putText(heatmap, f'Count: {count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # --- Spike alert overlay and action ---
    if spike_alert:
        cv2.rectangle(heatmap, (5, 60), (450, 110), (0, 0, 255), -1)
        cv2.putText(heatmap, 'EMERGENCY SPIKE DETECTED', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        play_beep()
        send_sms_alert("🚨 Emergency Alert: Crowd spike detected at Mahakumbh!")
        send_whatsapp_alert("🚨 Emergency Alert: Crowd spike detected at Mahakumbh!")

    cv2.imshow('Digital Twin - Crowd Heatmap', heatmap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
