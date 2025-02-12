import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
ROOT_DIR = r"D:/new researches/CAR RESEARCH/dataset 3/animals/output"  # Root directory
INPUT_FOLDERS = ["dog", "bear", "cats", "mouse", "wolf"]  # Subfolders containing audio files
OUTPUT_FEATURES_CSV = os.path.join(ROOT_DIR, "audio_features.csv")  # Path for saving extracted features
LOG_FILE = os.path.join(ROOT_DIR, "error_log.txt")  # Error log file
TARGET_SAMPLE_RATE = 16000  # Resampling rate (e.g., 16 kHz)

# Ensure the output folder exists
os.makedirs(ROOT_DIR, exist_ok=True)

def visualize_audio(file_path, y, sr):
    """
    Visualize the waveform and the Mel Spectrogram of an audio file.
    """
    plt.figure(figsize=(12, 6))

    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform: {os.path.basename(file_path)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot Mel Spectrogram
    plt.subplot(2, 1, 2)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis="time", y_axis="mel", cmap="viridis")
    plt.title("Mel Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()

    # Save visualization as an image
    visualization_path = os.path.join(ROOT_DIR, "visualizations")
    os.makedirs(visualization_path, exist_ok=True)
    output_file = os.path.join(visualization_path, f"{os.path.splitext(os.path.basename(file_path))[0]}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Visualization saved: {output_file}")

def extract_features(file_path):
    """
    Extract Mel Spectrogram, MFCCs, and Chromagram features from an audio file.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
        
        # Visualize the audio
        visualize_audio(file_path, y, sr)
        
        # Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13 coefficients is standard
        
        # Chromagram
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Mean and Standard Deviation for compact representation
        features = {
            "mel_spectrogram_mean": np.mean(mel_spectrogram_db),
            "mel_spectrogram_std": np.std(mel_spectrogram_db),
            "mfcc_mean": np.mean(mfccs, axis=1).tolist(),  # Per coefficient
            "mfcc_std": np.std(mfccs, axis=1).tolist(),
            "chromagram_mean": np.mean(chromagram, axis=1).tolist(),
            "chromagram_std": np.std(chromagram, axis=1).tolist()
        }
        
        return features
    except Exception as e:
        # Log the error
        with open(LOG_FILE, "a") as log:
            log.write(f"Error processing {file_path}: {e}\n")
        print(f"Error processing {file_path}: {e}")
        return None

# Process all files and extract features
all_features = []
for folder in INPUT_FOLDERS:
    input_folder = os.path.join(ROOT_DIR, folder)
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        if os.path.isfile(file_path) and file_name.lower().endswith((".wav", ".m4a")):
            print(f"Processing {file_path}...")
            features = extract_features(file_path)
            if features:
                features["label"] = folder  # Add label (e.g., 'dog', 'bear')
                features["file_name"] = file_name
                all_features.append(features)

# Save extracted features to CSV
if all_features:
    # Flatten nested lists for MFCCs and Chromagram stats
    flat_features = []
    for f in all_features:
        flat = {
            "file_name": f["file_name"],
            "label": f["label"],
            "mel_spectrogram_mean": f["mel_spectrogram_mean"],
            "mel_spectrogram_std": f["mel_spectrogram_std"]
        }
        flat.update({f"mfcc_mean_{i}": val for i, val in enumerate(f["mfcc_mean"])})
        flat.update({f"mfcc_std_{i}": val for i, val in enumerate(f["mfcc_std"])})
        flat.update({f"chromagram_mean_{i}": val for i, val in enumerate(f["chromagram_mean"])})
        flat.update({f"chromagram_std_{i}": val for i, val in enumerate(f["chromagram_std"])})
        flat_features.append(flat)
    
    # Save to CSV
    df = pd.DataFrame(flat_features)
    df.to_csv(OUTPUT_FEATURES_CSV, index=False)
    print(f"Features saved to {OUTPUT_FEATURES_CSV}")
else:
    print("No features extracted.")
