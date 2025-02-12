# Expanded features
import os
import librosa
import numpy as np
import pandas as pd

# Configuration
ROOT_DIR = r"D:\new researches\Car research2\dataset\DB1__segmented_audio"   # Root directory for input
OUTPUT_FEATURES_CSV = os.path.join(ROOT_DIR, "DB1_Categorized_features2.csv")  # Path for saving extracted features
TARGET_SAMPLE_RATE = 16000  # Resampling rate (16 kHz)

# Ensure output folder exists
os.makedirs(os.path.dirname(OUTPUT_FEATURES_CSV), exist_ok=True)

def extract_features(file_path):
    """
    Extract comprehensive audio features from a file.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
        
        # Extract features
        features = {}
        features["file_name"] = os.path.basename(file_path)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = np.mean(zcr)
        features["zcr_std"] = np.std(zcr)
        
        # Root Mean Square Energy
        rmse = librosa.feature.rms(y=y)
        features["rmse_mean"] = np.mean(rmse)
        features["rmse_std"] = np.std(rmse)
        
        # Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid_mean"] = np.mean(spectral_centroid)
        features["spectral_centroid_std"] = np.std(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
        features["spectral_bandwidth_std"] = np.std(spectral_bandwidth)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features["spectral_contrast_mean"] = np.mean(spectral_contrast, axis=1).tolist()
        features["spectral_contrast_std"] = np.std(spectral_contrast, axis=1).tolist()
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
        features["spectral_rolloff_std"] = np.std(spectral_rolloff)
        
        # Chroma Features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features["chroma_mean"] = np.mean(chroma_stft, axis=1).tolist()
        features["chroma_std"] = np.std(chroma_stft, axis=1).tolist()
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
        features["mfcc_std"] = np.std(mfccs, axis=1).tolist()
        
        # MFCC Delta
        mfcc_delta = librosa.feature.delta(mfccs)
        features["mfcc_delta_mean"] = np.mean(mfcc_delta, axis=1).tolist()
        features["mfcc_delta_std"] = np.std(mfcc_delta, axis=1).tolist()
        
        # MFCC Delta-Delta
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        features["mfcc_delta2_mean"] = np.mean(mfcc_delta2, axis=1).tolist()
        features["mfcc_delta2_std"] = np.std(mfcc_delta2, axis=1).tolist()
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process all files in the root directory
all_features = []
for folder in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith((".wav", ".m4a", ".mp3")):
                print(f"Processing file: {file_path}")
                features = extract_features(file_path)
                if features:
                    features["label"] = folder  # Use the folder name as the label
                    all_features.append(features)

# Save features to CSV
if all_features:
    # Flatten nested lists for Spectral Contrast, Chroma, and MFCC stats
    flat_features = []
    for f in all_features:
        flat = {
            "file_name": f["file_name"],
            "label": f["label"],
            "zcr_mean": f["zcr_mean"],
            "zcr_std": f["zcr_std"],
            "rmse_mean": f["rmse_mean"],
            "rmse_std": f["rmse_std"],
            "spectral_centroid_mean": f["spectral_centroid_mean"],
            "spectral_centroid_std": f["spectral_centroid_std"],
            "spectral_bandwidth_mean": f["spectral_bandwidth_mean"],
            "spectral_bandwidth_std": f["spectral_bandwidth_std"],
            "spectral_rolloff_mean": f["spectral_rolloff_mean"],
            "spectral_rolloff_std": f["spectral_rolloff_std"]
        }
        # Extend spectral contrast
        flat.update({f"spectral_contrast_mean_{i}": val for i, val in enumerate(f["spectral_contrast_mean"])})
        flat.update({f"spectral_contrast_std_{i}": val for i, val in enumerate(f["spectral_contrast_std"])})
        # Extend chroma features
        flat.update({f"chroma_mean_{i}": val for i, val in enumerate(f["chroma_mean"])})
        flat.update({f"chroma_std_{i}": val for i, val in enumerate(f["chroma_std"])})
        # Extend MFCCs and deltas
        flat.update({f"mfcc_mean_{i}": val for i, val in enumerate(f["mfcc_mean"])})
        flat.update({f"mfcc_std_{i}": val for i, val in enumerate(f["mfcc_std"])})
        flat.update({f"mfcc_delta_mean_{i}": val for i, val in enumerate(f["mfcc_delta_mean"])})
        flat.update({f"mfcc_delta_std_{i}": val for i, val in enumerate(f["mfcc_delta_std"])})
        flat.update({f"mfcc_delta2_mean_{i}": val for i, val in enumerate(f["mfcc_delta2_mean"])})
        flat.update({f"mfcc_delta2_std_{i}": val for i, val in enumerate(f["mfcc_delta2_std"])})
        flat_features.append(flat)
    
    # Save to CSV
    df = pd.DataFrame(flat_features)
    df.to_csv(OUTPUT_FEATURES_CSV, index=False)
    print(f"Features saved to {OUTPUT_FEATURES_CSV}")
else:
    print("No features extracted.")
