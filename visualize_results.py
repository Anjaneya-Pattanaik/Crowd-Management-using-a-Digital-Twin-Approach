import os
import numpy as np
import scipy.io as sio
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from model import CSRNet

# Paths
img_path = 'path to test_data/images'
gt_path = 'path to test_data/ground-truth'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CSRNet().to(device)
checkpoint = torch.load('path to PartAmodel_best.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((768, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create Gaussian kernel for GT density map generation
def generate_gt_density_map(image_shape, points):
    density_map = np.zeros(image_shape[:2], dtype=np.float32)
    for point in points:
        x, y = int(point[0]), int(point[1])
        if x >= image_shape[1] or y >= image_shape[0]:
            continue
        density_map[y, x] = 1
    density_map = cv2.GaussianBlur(density_map, (15, 15), 0)
    return density_map

# Get first 10 image filenames
image_files = sorted(os.listdir(img_path))[:10]

# Plot results
for i, img_file in enumerate(image_files):
    img_full_path = os.path.join(img_path, img_file)
    mat_file = os.path.join(gt_path, f'GT_{img_file.replace(".jpg", ".mat")}')

    # Load image
    orig_img = cv2.imread(img_full_path)
    display_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Preprocess for model
    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        density_map = model(input_tensor).squeeze().cpu().numpy()

    # Load ground truth points
    mat = sio.loadmat(mat_file)
    gt_points = mat['image_info'][0][0][0][0][0]
    gt_density_map = generate_gt_density_map(orig_img.shape, gt_points)

    # Normalize maps for visualization
    pred_vis = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
    gt_vis = cv2.normalize(gt_density_map, None, 0, 255, cv2.NORM_MINMAX)

    # Show figure
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(display_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_vis, cmap='jet')
    plt.title(f'Predicted Density (Count: {density_map.sum():.2f})')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gt_vis, cmap='jet')
    plt.title(f'Ground Truth (Count: {len(gt_points)})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'visual_result_{i+1}.png')  # Save image
    plt.show()
