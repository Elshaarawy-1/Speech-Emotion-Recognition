import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd

class Plotter:
    def __init__(self, sr=22050):
        self.sr = sr  # Sampling rate

    def plot_waveform(self, y, sr, title="Waveform"):
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    def plot_waveform_and_play(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sr)
        self.plot_waveform(y, sr, title=f"Waveform: {file_path}")
        print("Playing audio...")
        sd.play(y, sr)
        sd.wait()
    """
    def plot_waveform_and_play_ipython(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sr)
        self.plot_waveform(y, sr, title=f"Waveform: {file_path}")
        return Audio(y, rate=sr)
    """
    
