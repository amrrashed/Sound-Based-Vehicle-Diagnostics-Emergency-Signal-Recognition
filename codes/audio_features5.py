# NEW Compact Features Extraction Script
import os
import librosa
import numpy as np
import pandas as pd

#Key Changes:
# Feature Set: Added computation for RMS energy, spectral features (centroid, bandwidth, flux, roll-off), ZCR, pitch (F0), and harmonic-to-noise ratio (HNR).
# MFCC Enhancements: Included Delta and Delta-Delta MFCCs for temporal dynamics.
# Feature Flattening: Adjusted how features are flattened for CSV output.
# Error Handling: Improved robustness to handle edge cases (e.g., files with no pitch).

# Configuration
ROOT_DIR = r"D:\new researches\Car research2\dataset\DB1__segmented_audio"   # Root directory for input
OUTPUT_FEATURES_CSV = os.path.join(ROOT_DIR, "DB1_Categorized_features3.csv")  # Path for saving extracted features
TARGET_SAMPLE_RATE = 16000  # Resampling rate (16 kHz)

def extract_features(file_path):
    """
    Extract specified features from an audio file.
    Features:
    - MFCCs (13 coefficients + Delta/Delta-Delta)
    - RMS Energy
    - Spectral Centroid, Bandwidth, Flux, Roll-Off
    - Zero Crossing Rate
    - Pitch (F0)
    - Harmonic-to-Noise Ratio (HNR)
    """
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        
        # Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        
        # Pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0  # Average pitch
        
        # Harmonic-to-Noise Ratio (HNR)
        harmonic = librosa.effects.harmonic(y)
        hnr = np.mean(harmonic / (y + 1e-6))  # Avoid division by zero
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Aggregate all features into a dictionary
        features = {
            "rms_mean": np.mean(rms),
            "rms_std": np.std(rms),
            "spectral_centroid_mean": np.mean(spectral_centroid),
            "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
            "spectral_flux_mean": np.mean(spectral_flux),
            "spectral_rolloff_mean": np.mean(spectral_rolloff),
            "zcr_mean": np.mean(zcr),
            "pitch_mean": pitch,
            "hnr_mean": hnr,
            "mfcc_mean": np.mean(mfccs, axis=1).tolist(),
            "mfcc_std": np.std(mfccs, axis=1).tolist(),
            "delta_mfcc_mean": np.mean(delta_mfccs, axis=1).tolist(),
            "delta2_mfcc_mean": np.mean(delta2_mfccs, axis=1).tolist(),
        }
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Automatically read all subfolders in the root directory
input_folders = [f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))]

# Process all files and extract features
all_features = []
for folder in input_folders:
    input_folder = os.path.join(ROOT_DIR, folder)
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        if os.path.isfile(file_path) and file_name.lower().endswith((".wav", ".m4a", ".mp3")):
            print(f"Processing {file_path}...")
            features = extract_features(file_path)
            if features:
                features["label"] = folder  # Add label (e.g., subfolder name: 'dog', 'car_fault', etc.)
                features["file_name"] = file_name
                all_features.append(features)

# Save extracted features to CSV
if all_features:
    # Flatten nested lists for MFCCs, Delta MFCCs, etc.
    flat_features = []
    for f in all_features:
        flat = {
            "file_name": f["file_name"],
            "label": f["label"],
            "rms_mean": f["rms_mean"],
            "rms_std": f["rms_std"],
            "spectral_centroid_mean": f["spectral_centroid_mean"],
            "spectral_bandwidth_mean": f["spectral_bandwidth_mean"],
            "spectral_flux_mean": f["spectral_flux_mean"],
            "spectral_rolloff_mean": f["spectral_rolloff_mean"],
            "zcr_mean": f["zcr_mean"],
            "pitch_mean": f["pitch_mean"],
            "hnr_mean": f["hnr_mean"]
        }
        flat.update({f"mfcc_mean_{i}": val for i, val in enumerate(f["mfcc_mean"])})
        flat.update({f"mfcc_std_{i}": val for i, val in enumerate(f["mfcc_std"])})
        flat.update({f"delta_mfcc_mean_{i}": val for i, val in enumerate(f["delta_mfcc_mean"])})
        flat.update({f"delta2_mfcc_mean_{i}": val for i, val in enumerate(f["delta2_mfcc_mean"])})
        flat_features.append(flat)
    
    # Save to CSV
    df = pd.DataFrame(flat_features)
    df.to_csv(OUTPUT_FEATURES_CSV, index=False)
    print(f"Features saved to {OUTPUT_FEATURES_CSV}")
else:
    print("No features extracted.")
