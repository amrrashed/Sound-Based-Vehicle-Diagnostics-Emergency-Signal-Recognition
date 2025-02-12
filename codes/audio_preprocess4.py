import os
import csv
from pydub import AudioSegment
import librosa

# Configuration
ROOT_DIR = r"D:\new researches\CAR RESEARCH\datasets\DB1"  # Root directory for input
EXTENDED_FOLDER = r"D:\new researches\CAR RESEARCH\datasets\DB1_extended audio files"  # Folder for extended audio
SEGMENTED_FOLDER = r"D:\new researches\CAR RESEARCH\datasets\DB1_segmented_audio"  # Folder for segmented audio
TARGET_SAMPLE_RATE = 16000  # Resampling rate (16 kHz)
SEGMENT_DURATION_MS = 2500  # Segment length in milliseconds (e.g., 2.5 seconds)
MIN_DURATION_MS = 10000  # Minimum duration for an audio file (10 seconds)

# Ensure output folders exist
os.makedirs(EXTENDED_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)

def extend_audio_to_minimum_duration(audio, min_duration_ms):
    """
    Extend the audio by repeating it until it reaches the minimum duration.
    """
    if len(audio) < min_duration_ms:
        repeat_count = (min_duration_ms // len(audio)) + 1  # Calculate the number of repetitions needed
        audio = audio * repeat_count
        audio = audio[:min_duration_ms]  # Trim to exact duration
    return audio

def preprocess_audio(file_path, extended_folder, segmented_folder):
    """
    Normalize, resample, save extended audio, and segment the audio file.
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        
        # Extend audio if it is shorter than the minimum duration
        extended_audio = extend_audio_to_minimum_duration(audio, MIN_DURATION_MS)
        
        # Normalize audio
        normalized_audio = extended_audio.normalize()
        
        # Save the extended and normalized audio
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        extended_audio_path = os.path.join(extended_folder, f"{base_name}_extended.wav")
        normalized_audio.export(extended_audio_path, format="wav")
        print(f"Saved extended audio: {extended_audio_path}")
        
        # Resample the extended audio
        y, sr = librosa.load(extended_audio_path, sr=TARGET_SAMPLE_RATE)
        resampled_audio = AudioSegment(
            y.tobytes(),
            frame_rate=TARGET_SAMPLE_RATE,
            sample_width=normalized_audio.sample_width,
            channels=len(normalized_audio.split_to_mono())
        )
        
        # Segment the resampled audio
        duration_ms = len(resampled_audio)
        for i in range(0, duration_ms, SEGMENT_DURATION_MS):
            segment = resampled_audio[i:i + SEGMENT_DURATION_MS]
            if len(segment) < SEGMENT_DURATION_MS:
                continue  # Skip segments that are too short
            
            # Save each segment
            segment_output_path = os.path.join(segmented_folder, f"{base_name}_segment_{i // SEGMENT_DURATION_MS}.wav")
            segment.export(segment_output_path, format="wav")
            print(f"Saved segment: {segment_output_path}")
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Automatically read all subfolders in the root directory
input_folders = [f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))]

# Process files in each subfolder
for folder in input_folders:
    input_folder = os.path.join(ROOT_DIR, folder)
    extended_subfolder = os.path.join(EXTENDED_FOLDER, folder)
    segmented_subfolder = os.path.join(SEGMENTED_FOLDER, folder)
    
    # Ensure subfolders exist for extended and segmented audio
    os.makedirs(extended_subfolder, exist_ok=True)
    os.makedirs(segmented_subfolder, exist_ok=True)
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        if os.path.isfile(file_path) and file_name.lower().endswith((".wav", ".m4a", ".mp3")):
            print(f"Processing file: {file_path}")
            preprocess_audio(file_path, extended_subfolder, segmented_subfolder)

# Count the number of instances for each label in the segmented folder
label_counts = {}

# Walk through all files in the segmented folder and count instances based on folder names or file names
for root, dirs, files in os.walk(SEGMENTED_FOLDER):
    for file in files:
        # Extract label from the folder name or file name
        label = os.path.basename(root)  # Assuming the label is the name of the folder
        # Alternatively, you can extract the label from the file name if needed:
        # label = file.split('_')[0]  # Assuming label is part of the file name
        
        # Increment the count for the label
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

# Save the label counts to a CSV file
csv_file_path = r"D:\new researches\CAR RESEARCH\datasets\label_counts_db1.csv"

# Write the label counts to CSV
with open(csv_file_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Label", "Count"])  # Header
    for label, count in label_counts.items():
        writer.writerow([label, count])

print(f"Label counts saved to {csv_file_path}")
print("Preprocessing complete!")
