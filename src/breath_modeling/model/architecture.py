import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    """
    Simple CNN classifier for 1D time series data.
    Assumes input shape [B, T, C] where B is batch size, T is time steps, and C is channels.
    The model consists of two convolutional layers followed by max pooling, dropout, and two fully connected layers.
    The output layer has n_classes outputs.
    The input_len parameter should be the length of the time series input.
    The n_classes parameter specifies the number of output classes for classification.
    The model expects input with a single channel (C=1), so input data should be reshaped accordingly.
    """
    def __init__(self, input_len, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * (input_len // 4), 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C=1, T]
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class CNNBiLSTMClassifier(nn.Module):
    """
    CNN + BiLSTM classifier for 1D time series data.
    Combines convolutional layers for feature extraction with a bidirectional LSTM for sequence modeling.
    The model expects input shape [B, S, T, C] where B is batch size, S is the number of sequences (e.g., segments), T is time steps, and C is channels.
    The CNN layers extract features from each sequence, which are then processed by a bidirectional LSTM.
    The output layer has n_classes outputs.
    The window_len parameter should be the length of each sequence, and n_classes specifies the number of output classes for classification.
    The cnn_out_channels parameter controls the number of output channels from the CNN layers, and lstm_hidden specifies the hidden size of the LSTM.
    """
    def __init__(self, window_len, n_classes, cnn_out_channels=64, lstm_hidden=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, cnn_out_channels, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        cnn_out_dim = cnn_out_channels * (window_len // 4)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(2 * lstm_hidden, n_classes)

    def forward(self, x):
        B, S, T, C = x.shape
        x = x.view(B * S, C, T)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(B, S, -1)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        return self.classifier(lstm_out)  # [B, S, n_classes]