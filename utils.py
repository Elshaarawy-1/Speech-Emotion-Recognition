import os
from SpeechModel import FirstModel, SecondModel
from torch.utils.data import DataLoader
from dataset import AudioEmotionDataset
from sklearn.model_selection import train_test_split

EMOTION_DICT_CREMA = {
    "NEU": "neutral",
    "SAD": "sadness",
    "HAP": "happy",
    "ANG": "angry",
    "FEA": "fear",
    "DIS": "disgust",
}

def get_dataset(
    training_dir="./data/Crema-processed",
    label_dict=EMOTION_DICT_CREMA,
    batch_size=64,
    random_state=42,
    num_workers=4,
):
    """
    Returns PyTorch DataLoaders for train, val, test splits with stratification.
    """
    label_to_int = {key: i for i, key in enumerate(label_dict.keys())}

    def decompose_label(file_path: str):
        return label_to_int[file_path.split("_")[2]]

    # Get all file paths and corresponding labels
    file_names = os.listdir(training_dir)
    file_paths = [os.path.join(training_dir, f) for f in file_names]
    labels = [decompose_label(f) for f in file_names]

    # Stratified split: test 30%, temp 70%
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.3, stratify=labels, random_state=random_state
    )

    # Stratified split: train 95%, val 5% from temp
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=0.05,
        stratify=train_val_labels,
        random_state=random_state,
    )

    # Create datasets
    train_dataset = AudioEmotionDataset(train_paths, train_labels)
    val_dataset = AudioEmotionDataset(val_paths, val_labels)
    test_dataset = AudioEmotionDataset(test_paths, test_labels)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def create_model(model_type="1"):
    model_type = model_type.lower()
    if model_type == "1":
        model = FirstModel()
    elif model_type == "2":
        model = SecondModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model
