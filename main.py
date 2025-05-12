import pandas as pd

from load_dataset.loader import Loader
from load_dataset.plotter import Plotter
def main():
    crema_path = "Dataset/CREMA/"  # replace with your actual path
    data_loader = Loader()
    data_loader.load_data(crema_path)

    CREMA_df = pd.DataFrame(data_loader.emotionG, columns=['emotionG_label'])
    CREMA_df['source'] = 'CREMA'
    CREMA_df['gender'] = data_loader.gender
    CREMA_df['emotion'] = data_loader.emotionO
    CREMA_df['path'] = data_loader.path

    plotter = Plotter()
    emotions_seen = set()
    for i in range(len(CREMA_df)):
        emotion = CREMA_df.loc[i, 'emotion']
        if emotion not in emotions_seen:
            print(f"\nEmotion: {emotion}")
            plotter.plot_waveform_and_play(CREMA_df.loc[i, 'path'])
            emotions_seen.add(emotion)
        if len(emotions_seen) == len(CREMA_df['emotion'].unique()):
            break

if __name__ == "__main__":
    main()

    
