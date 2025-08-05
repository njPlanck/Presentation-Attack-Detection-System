import os
import cv2
import numpy as np
import pandas as pd

# Compute Fourier-based band energy features
def feature_vector(image, num_bands=30, slice_range=(0, 30)):
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    energy_spectrum = np.abs(fft_shifted) ** 2

    rows, cols = image.shape
    center = (rows // 2, cols // 2)
    max_radius = np.sqrt(center[0] ** 2 + center[1] ** 2)
    band_width = max_radius / num_bands

    start_band, end_band = slice_range
    features = []

    for i in range(start_band, end_band):
        inner_radius = i * band_width
        outer_radius = (i + 1) * band_width
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        mask = (distance >= inner_radius) & (distance < outer_radius)
        band_energy = np.sum(energy_spectrum[mask])
        features.append(band_energy)

    return np.array(features)

# Load images from a folder and extract features
def load_images_and_extract_features(file_paths, label, num_bands=30, slice_range=(0, 30)):
    data = []

    for path in tqdm(file_paths):
        image = cv2.imread(path, 0)  # Load in grayscale
        if image is None:
            continue
        features = feature_vector(image, num_bands, slice_range)
        entry = {"file_path": path, "label": label}
        for i, feat in enumerate(features):
            entry[f"feature_{i}"] = feat
        data.append(entry)

    return data
# Main execution
if __name__ == "__main__":
    real_folder = glob.glob(r'images/genuine/scut/**/*.bmp', recursive=True)
    fake_folder = glob.glob(r'images/spoofed/scut_spoofed/**/*.bmp', recursive=True)
    synthetic_folder = glob.glob(r'mean_output_02/synthetic_attacked/scut/cycleGAN/*.png',recursive=True)  # Adjust path

    data_real = load_images_and_extract_features(real_folder, label=1)
    data_fake = load_images_and_extract_features(fake_folder, label=0)
    data_synthetic = load_images_and_extract_features(synthetic_folder, label=2)

    combined_data = data_real + data_fake + data_synthetic
    df = pd.DataFrame(combined_data)

    os.makedirs("output_csvs", exist_ok=True)
    df.to_csv("output_csvs/features_all.csv", index=False)

    print("DataFrame columns:", df.columns.tolist())
    print("First few rows:\n", df.head())

    df_subset1 = df[["label"] + [f"feature_{i}" for i in range(30)]]
    df_subset1.to_csv("output_csvs/scut_mean02/cycle_features_01_30.csv", index=False)

    df_subset2 = df[["label"] + [f"feature_{i}" for i in range(10)]]
    df_subset2.to_csv("output_csvs/scut_mean02/cycle_features_01_10.csv", index=False)

    df_subset3 = df[["label"] + [f"feature_{i}" for i in range(10, 20)]]
    df_subset3.to_csv("output_csvs/scut_mean02/cycle_features_11_20.csv", index=False)

    df_subset4 = df[["label"] + [f"feature_{i}" for i in range(20, 30)]]
    df_subset4.to_csv("output_csvs/scut_mean02/cycle_features_21_30.csv", index=False)

    print("Feature extraction and export completed.")