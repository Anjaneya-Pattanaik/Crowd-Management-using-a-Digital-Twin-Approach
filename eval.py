import os
import torch
import numpy as np
import scipy.io as sio
import cv2
import csv
from torchvision import transforms
from model import CSRNet
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CSRNet().to(device)
checkpoint = torch.load('path to PartAmodel_best.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
model.eval()

# Paths
image_dir = 'path to test_data/images'
mat_dir = 'path to test_data/ground-truth'

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((768, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mae, mse = 0.0, 0.0
psnr_total, ssim_total = 0.0, 0.0
pseudo_acc_count = 0
tolerance_ratio = 0.1
num_samples = 0
results_per_image = []

def generate_gt_density_map(image_shape, points):
    density_map = np.zeros(image_shape[:2], dtype=np.float32)
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            density_map[y, x] = 1
    density_map = cv2.GaussianBlur(density_map, (15, 15), 0)
    return density_map

for filename in os.listdir(image_dir):
    if not (filename.endswith('.jpg') or filename.endswith('.png')):
        continue

    img_path = os.path.join(image_dir, filename)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    mat_name = 'GT_' + filename.replace('.jpg', '.mat').replace('.png', '.mat')
    mat_path = os.path.join(mat_dir, mat_name)
    mat = sio.loadmat(mat_path)
    gt_points = mat['image_info'][0][0][0][0][0]
    gt_count = gt_points.shape[0]

    with torch.no_grad():
        output = model(input_tensor)
    pred_density = output.squeeze().cpu().numpy()
    pred_count = pred_density.sum().item()

    # Resize pred_density to original image size for PSNR/SSIM
    pred_density_resized = cv2.resize(pred_density, (img.shape[1], img.shape[0]))
    gt_density = generate_gt_density_map(img.shape, gt_points)

    # Metrics
    mae += abs(gt_count - pred_count)
    mse += (gt_count - pred_count) ** 2

    # PSNR & SSIM (clip values for stability)
    psnr_total += compare_psnr(gt_density, np.clip(pred_density_resized, 0, None), data_range=gt_density.max() - gt_density.min())
    ssim_total += compare_ssim(gt_density, np.clip(pred_density_resized, 0, None), data_range=gt_density.max() - gt_density.min())

    # Pseudo Accuracy
    tolerance = max(1, int(gt_count * tolerance_ratio))
    if abs(gt_count - pred_count) <= tolerance:
        pseudo_acc_count += 1

    num_samples += 1
    results_per_image.append([filename, gt_count, f'{pred_count:.2f}'])

# Final Results
mae /= num_samples
mse /= num_samples
psnr_avg = psnr_total / num_samples
ssim_avg = ssim_total / num_samples
pseudo_accuracy = 100.0 * pseudo_acc_count / num_samples

print('\n--- Evaluation Results ---')
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'Pseudo Accuracy (@{int(tolerance_ratio*100)}% tolerance): {pseudo_accuracy:.2f}%')

# Export to CSV
output_file = 'updated_evaluation_results.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Ground Truth Count', 'Predicted Count'])
    for row in results_per_image:
        writer.writerow(row)
    writer.writerow([])
    writer.writerow(['Overall MAE', f'{mae:.2f}'])
    writer.writerow(['Overall MSE', f'{mse:.2f}'])
    writer.writerow(['PSNR', f'{psnr_avg:.2f}'])
    writer.writerow(['SSIM', f'{ssim_avg:.4f}'])
    writer.writerow([f'Pseudo Accuracy (@{int(tolerance_ratio*100)}% tolerance)', f'{pseudo_accuracy:.2f}%'])

print(f'\n✅ Results exported to {output_file}')
