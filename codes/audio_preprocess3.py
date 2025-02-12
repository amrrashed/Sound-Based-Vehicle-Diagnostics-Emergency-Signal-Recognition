import os
from pydub import AudioSegment
import librosa

# Configuration
ROOT_DIR = r"D:\new researches\CAR RESEARCH\dataset 3\car problems"  # Root directory for input
OUTPUT_FOLDER = r"D:\new researches\CAR RESEARCH\dataset 3\output"  # Root directory for processed files
TARGET_SAMPLE_RATE = 16000  # Resampling rate (16 kHz)
SEGMENT_DURATION_MS = 2500  # Segment length in milliseconds (e.g., 2.5 seconds)
MIN_DURATION_MS = 10000  # Minimum duration for an audio file (10 seconds)

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extend_audio_to_minimum_duration(audio, min_duration_ms):
    """
    Extend the audio by repeating it until it reaches the minimum duration.
    """
    if len(audio) < min_duration_ms:
        repeat_count = (min_duration_ms // len(audio)) + 1  # Calculate the number of repetitions needed
        audio = audio * repeat_count
        audio = audio[:min_duration_ms]  # Trim to exact duration
    return audio

def preprocess_audio(file_path, output_folder):
    """
    Normalize, resample, and segment audio file, saving processed clips.
    """
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        
        # Extend audio if it is shorter than the minimum duration
        extended_audio = extend_audio_to_minimum_duration(audio, MIN_DURATION_MS)
        
        # Normalize audio
        normalized_audio = extended_audio.normalize()
        
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

# Automatically read all subfolders in the root directory
input_folders = [f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))]

# Process files in each subfolder
for folder in input_folders:
    input_folder = os.path.join(ROOT_DIR, folder)
    output_subfolder = os.path.join(OUTPUT_FOLDER, folder)
    os.makedirs(output_subfolder, exist_ok=True)
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        if os.path.isfile(file_path) and file_name.lower().endswith((".wav", ".m4a", ".mp3")):
            print(f"Processing file: {file_path}")
            preprocess_audio(file_path, output_subfolder)

print("Preprocessing complete!")
