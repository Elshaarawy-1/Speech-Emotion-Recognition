from glob import glob
import os
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def process_audio_clip(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path)

    # Extract features
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    # mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    chromagram = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    spectral = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=audio, sr=sr).T, axis=0)

    # Combine all features into one feature vector
    extracted_features = np.concatenate([mfcc, chromagram, spectral, tonnetz], axis=0)

    return extracted_features

def process_all_audio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    feature_shapes = set()

    for file_path in glob(os.path.join(input_dir, "*.wav")):
        features = process_audio_clip(file_path)
        feature_shapes.add(features.shape)

        file_name = os.path.splitext(os.path.basename(file_path))[0]
        np.save(os.path.join(output_dir, file_name + ".npy"), features)

    # Show one example and validate shapes
    print(f"Sample shape: {features.shape}")
    if len(feature_shapes) > 1:
        print("Inconsistent feature shapes found:", feature_shapes)
    else:
        print("All features have consistent shape:", feature_shapes.pop())


if __name__ == "__main__":
    input_dir = "./data/Crema"
    output_dir = "./data/Crema-processed"
    process_all_audio(input_dir, output_dir)