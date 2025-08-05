import numpy as np
import pandas as pd
import cv2
import glob
from skimage.feature import graycomatrix, graycoprops
import os

def extract_glcm_features(img):
    df = pd.DataFrame()

    for angle_id, angle in enumerate([0, np.pi/4, np.pi/2]):
        glcm = graycomatrix(img, [5], [angle])
        prefix = f"{'' if angle_id == 0 else angle_id+1}"

        df[f'Diss{prefix}'] = graycoprops(glcm, 'dissimilarity')[0]
        df[f'Energy{prefix}'] = graycoprops(glcm, 'energy')[0]
        df[f'Homo{prefix}'] = graycoprops(glcm, 'homogeneity')[0]
        df[f'Corr{prefix}'] = graycoprops(glcm, 'correlation')[0]
        df[f'ASM{prefix}'] = graycoprops(glcm, 'ASM')[0]
        df[f'Contrast{prefix}'] = graycoprops(glcm, 'contrast')[0]

    return df

def load_images_and_extract_features(image_paths, label):
    features = pd.DataFrame()
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        df = extract_glcm_features(img)
        df["label"] = label
        features = pd.concat([features, df], ignore_index=True)
    return features

if __name__ == "__main__":
    # Adjust glob patterns to match your structure
    real_folder = glob.glob(r'images/genuine/scut/**/*.bmp', recursive=True)
    fake_folder = glob.glob(r'images/spoofed/scut_spoofed/**/*.bmp', recursive=True)
    synthetic_folder = glob.glob(r'mean_output/synthetic_attacked/scut/cycleGAN/*.png', recursive=True)

    # Extract features for each group
    data_real = load_images_and_extract_features(real_folder, label=1)
    data_fake = load_images_and_extract_features(fake_folder, label=0)
    data_synthetic = load_images_and_extract_features(synthetic_folder, label=2)

    # Combine all features into one dataset
    all_data = pd.concat([data_real, data_fake, data_synthetic], ignore_index=True)
    all_data.to_csv("haralick_csvs/scut/mean/cycle_features.csv", index=False)
