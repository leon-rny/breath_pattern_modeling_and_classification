import os
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .dataset import BreathDataset, BreathSequenceDataset
from .config import *

def load_dataset(task="cnn", max_samples_per_pattern=50, batch_size=32, test_size=0.2, random_state=42):
    """
    Loads the dataset for the specified task.

    :param task: The type of task to load the dataset for. Options are "random_forest", "cnn", "lstm", or "lstm_seq".
    :param max_samples_per_pattern: Maximum number of samples per pattern for the random forest task.
    :param batch_size: Batch size for the data loaders (used in CNN and LSTM tasks).
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.
    """
    if task == "random_forest":
        return _load_rf_dataset(max_samples_per_pattern)
    elif task == "cnn":
        return _load_cnn_dataset(batch_size, test_size, random_state)
    elif task == "lstm":
        return _load_lstm_dataset(batch_size, test_size, random_state)
    elif task == "lstm_seq":
        return _load_lstm_sequence_level(batch_size, test_size, random_state)
    else:
        raise ValueError(f"Unknown task: {task}")

def _load_rf_dataset(max_samples_per_pattern):
    """
    Loads the dataset for the random forest task.
    :param max_samples_per_pattern: Maximum number of samples per pattern to include in the dataset.
    :return: A DataFrame containing the long format of the dataset.
    """
    with open(SAMPLE_METADATA_FILE, "r") as f:
        metadata = [json.loads(line) for line in f]

    data_long = []
    pattern_counter = {}

    # loop through metadata and load samples
    for meta in metadata:
        pattern = meta["pattern"]
        idx = meta["sample_index"]
        sample_id = f"{pattern}_{idx}"
        # Check if we have already reached the max samples for this pattern
        if pattern_counter.get(pattern, 0) >= max_samples_per_pattern:
            continue

        path = os.path.join(SAMPLES_DIR, f"{sample_id}.npy")
        if not os.path.exists(path):
            continue
        # Load the signal data
        signal = np.load(path)
        for t, val in enumerate(signal):
            data_long.append({"id": sample_id, "time": t, "value": val, "label": pattern})

        pattern_counter[pattern] = pattern_counter.get(pattern, 0) + 1

    df_long = pd.DataFrame(data_long)
    return df_long 

def _load_cnn_dataset(batch_size, test_size, random_state):
    """
    Loads the dataset for the CNN task.
    :param batch_size: Batch size for the data loaders.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.
    :return: Data loaders for training and validation sets, and a label encoder.
    """
    # Load the dataset from the JSON file
    df = pd.read_json(WINDOW_LABELS_FILE, lines=True)
    X = np.stack(df["signal"].values).astype(np.float32)[:, :, np.newaxis]
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    y = torch.tensor(y, dtype=torch.long)

    # create tensors for the signals
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Create datasets
    train_ds = BreathDataset(X_train, y_train)
    val_ds = BreathDataset(X_val, y_val)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, le

def _load_lstm_dataset(batch_size, test_size, random_state):
    """
    Loads the dataset for the LSTM task.
    :param batch_size: Batch size for the data loaders.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.
    """
    # Load the dataset from the JSON file
    df = pd.read_json(SEQUENCE_WINDOW_LABELS_FILE, lines=True)
    X = [np.array(seq["windows"], dtype=np.float32) for _, seq in df.iterrows()]
    y_raw = [seq["labels"] for _, seq in df.iterrows()]

    le = LabelEncoder()
    le.fit([label for seq in y_raw for label in seq])
    y = [[le.transform([label])[0] for label in seq] for seq in y_raw]

    # Convert to tensors
    X = [torch.tensor(seq[:, :, np.newaxis], dtype=torch.float32) for seq in X]
    y = [torch.tensor(seq, dtype=torch.long) for seq in y]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create datasets
    train_ds = BreathSequenceDataset(X_train, y_train)
    val_ds = BreathSequenceDataset(X_val, y_val)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, le

def _load_lstm_sequence_level(batch_size, test_size, random_state):
    """
    Loads the dataset for the LSTM sequence level task.
    :param batch_size: Batch size for the data loaders.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.
    :return: Data loaders for training and validation sets, and a label encoder."""
    # Load the dataset from the JSON file
    df = pd.read_json(SEQUENCE_WINDOW_LABELS_FILE, lines=True)

    X = [np.array(seq["windows"], dtype=np.float32) for _, seq in df.iterrows()]
    y_raw_nested = [seq["labels"] for _, seq in df.iterrows()] 

    flat_labels = [label for labels in y_raw_nested for label in labels]
    le = LabelEncoder()
    le.fit(flat_labels)

    # Convert labels to indices
    y = [max(set(le.transform(labels)), key=list(le.transform(labels)).count) for labels in y_raw_nested]
    y = torch.tensor(y, dtype=torch.long)
    X = [torch.tensor(seq[:, :, np.newaxis], dtype=torch.float32) for seq in X]

    # data split
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)

    # Create datasets
    train_ds = BreathSequenceDataset(X_train, y_train)
    val_ds = BreathSequenceDataset(X_val, y_val)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, le