import os
from pydub import AudioSegment
import librosa

# Configuration
ROOT_DIR = r"D:\new researches\CAR RESEARCH\dataset 3\animals"  # Root directory
INPUT_FOLDERS = ["dog", "bear" , "cats", "mouse" ,"wolf"]  # Subfolders containing audio files
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "output")  # Root folder for processed files
TARGET_SAMPLE_RATE = 16000      # Resampling rate (16 kHz)
SEGMENT_DURATION_MS = 2500      # Segment length in milliseconds (e.g., 2.5 seconds)

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_audio(file_path, output_folder):
    """
    Normalize, resample, and segment audio file, saving processed clips.
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        
        # Normalize audio
        normalized_audio = audio.normalize()
        
        # Resample audio
        y, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
        resampled_audio = AudioSegment(
            y.tobytes(),
            frame_rate=TARGET_SAMPLE_RATE,
            sample_width=audio.sample_width,
            channels=len(audio.split_to_mono())
        )
        
        # Segment audio
        duration_ms = len(resampled_audio)
        for i in range(0, duration_ms, SEGMENT_DURATION_MS):
            segment = resampled_audio[i:i + SEGMENT_DURATION_MS]
            if len(segment) < SEGMENT_DURATION_MS:
                continue  # Skip segments that are too short
            
            # Save each segment
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(output_folder, f"{base_name}_segment_{i // SEGMENT_DURATION_MS}.wav")
            segment.export(output_file, format="wav")
            print(f"Saved segment: {output_file}")
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Process files in each input folder
for folder in INPUT_FOLDERS:
    input_folder = os.path.join(ROOT_DIR, folder)
    output_subfolder = os.path.join(OUTPUT_FOLDER, folder)
    os.makedirs(output_subfolder, exist_ok=True)
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        if os.path.isfile(file_path) and file_name.lower().endswith((".wav", ".m4a", ".mp3")):
            print(f"Processing file: {file_path}")
            preprocess_audio(file_path, output_subfolder)

print("Preprocessing complete!")
