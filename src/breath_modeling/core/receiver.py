import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

from breath_modeling.model.architecture import CNNClassifier, CNNBiLSTMClassifier

class Receiver:
    """
    Simulates a signal receiver with various noise and distortion characteristics.

    :param sampling_rate (int): Sampling rate in Hz (default is 1000)
    :param amplitude (float): Amplitude of the signal (default is 1.0)
    :param model_type (str): Type of model to use for classification ('rf', 'cnn', 'bilstm')
    """
    def __init__(self, sampling_rate=1000, model_type=None):
        self.sampling_rate = sampling_rate
        self.amplitude = 1.0
        self.model_type = model_type  # 'rf', 'cnn', 'bilstm'
        self.model_path = Path(__file__).resolve().parents[3] / "src" /  "breath_modeling" / "core" / "models" / self.model_type
        self.model = None
        self.label_encoder = None
        self.selected_features = None
        self.model_config = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type:
            self.load_model()

    def load_model(self):
        """
        Loads the model and its associated artifacts from disk.
        """
        # ML
        if self.model_type == 'rf':
            data = joblib.load(self.model_path / "random_forest.joblib")
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            self.selected_features = data['selected_features']

        # DL
        else:
            with open(self.model_path / "label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)

            with open(self.model_path / "config.json", "r") as f:
                self.model_config = json.load(f)

            self.model = self._load_dl_model()
            self.model.load_state_dict(torch.load(self.model_path / "model.pth"))
            self.model.to(self.device)
            self.model.eval()

    def _load_dl_model(self):
        """
        Loads the deep learning model architecture based on the specified model type.
        :return: Initialized model instance
        :raises ValueError: If the model type is not supported
        """
        if self.model_type == "cnn":
            return CNNClassifier(input_len=self.model_config["input_len"], n_classes=len(self.label_encoder.classes_))

        elif self.model_type == "bilstm":
            return CNNBiLSTMClassifier(window_len=self.model_config["window_len"], n_classes=len(self.label_encoder.classes_))

        else:
            raise ValueError(f"Unsupported DL model type: {self.model_type}")

    def predict(self, X):
        """
        Predicts class labels for the given input data.

        :param X: 2D numpy array of shape (n_samples, n_features)
        :return: Predicted class labels as a list of strings
        """
        if self.model_type == 'rf':
            X_selected = X[self.selected_features]
            y_pred = self.model.predict(X_selected)
            return self.label_encoder.inverse_transform(y_pred)

        else:  # DL
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.model(X_tensor)
                y_pred = logits.argmax(dim=1).cpu().numpy()
                return self.label_encoder.inverse_transform(y_pred)

    def predict_signal_rf(self, signal, fs=1000):
        """
        Extract features from raw signal and classify with Random Forest.

        :param signal: 1D numpy array
        :param fs: sampling frequency in Hz
        :return: predicted label as string
        """
        if self.model_type != "rf":
            raise ValueError("Only valid for Random Forest models.")

        # time stamp
        signal = np.asarray(signal)
        df = pd.DataFrame({
            "id": 0,
            "time": np.arange(len(signal)) / fs,
            "value": signal
        })

        # feature extraction
        features = extract_features(
            df,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=EfficientFCParameters(),
            disable_progressbar=True,
        )

        # impute missing values
        features = impute(features)

        # select features
        X_selected = features[self.selected_features]

        # predict
        pred = self.model.predict(X_selected)[0]
        label = self.label_encoder.inverse_transform([pred])[0]
        return label

    def predict_window(self, signal, fs=500, window_size=3.0, stride=1.5):
        """
        Predicts class labels over a sliding window for a given signal.

        :param signal: 1D numpy array of input signal
        :param fs: Sampling rate (Hz)
        :param window_size: Window size in seconds
        :param stride: Stride size in seconds
        :return: (pred_labels, starts_in_seconds)
        """
        signal = np.asarray(signal)
        win_len = int(window_size * fs)
        stride_len = int(stride * fs)

        windows = []
        starts = []

        for start_idx in range(0, len(signal) - win_len + 1, stride_len):
            window = signal[start_idx: start_idx + win_len]
            windows.append(window)
            starts.append(start_idx / fs)

        X = np.array(windows)[..., np.newaxis]  # [S, T, 1]

        if self.model_type in ["cnn"]:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits = self.model(X_tensor)
                preds = logits.argmax(dim=-1).cpu().numpy()
                pred_labels = self.label_encoder.inverse_transform(preds)

        elif self.model_type == "bilstm":
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, S, T, 1]
            with torch.no_grad():
                logits = self.model(X_tensor)[0]  # [S, C]
                preds = logits.argmax(dim=-1).cpu().numpy()
                pred_labels = self.label_encoder.inverse_transform(preds)

        else:
            raise NotImplementedError(f"Sliding prediction not implemented for model type: {self.model_type}")

        return pred_labels, starts
    
    # receiver noise
    def add_noise(self, signal, noise_type='salt_pepper', **kwargs):
        """
        Adds noise to the signal based on the specified type using utils.noise module.

        :param signal: input signal
        :param noise_type: type of noise ('salt_pepper', 'quantization')
        :param kwargs: parameters passed to the noise function
        :return: noisy signal
        """
        signal = np.asarray(signal)

        if noise_type == 'salt_pepper':
            prob = kwargs.get('prob', 0.01)
            amplitude = kwargs.get('amplitude', 1.0)

            noisy = np.copy(signal)
            rnd = np.random.rand(*signal.shape)

            # Salt: ampliude
            salt_mask = rnd < (prob / 2)
            noisy[salt_mask] = amplitude

            # Pepper: 0
            pepper_mask = (rnd >= (prob / 2)) & (rnd < prob)
            noisy[pepper_mask] = 0.0

            return noisy

        elif noise_type == 'quantization':
            levels = kwargs.get('levels', 16)
            min_val, max_val = np.min(signal), np.max(signal)
            step = (max_val - min_val) / (levels - 1)
            quantized = np.round((signal - min_val) / step) * step + min_val
            return quantized

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    def baseline_drift(self, t, drift_type='linear', **kwargs):
        """
        Simulates baseline drift over time.

        :param t (float): Current time in seconds.
        :param drift_type (str): Type of drift ('linear', 'exponential', 'sinusoidal').
        :param kwargs: Parameters like rate, amplitude, etc.
        :return: Drift value at time t.
        """
        if drift_type == 'linear':
            rate = kwargs.get('rate', 0.01)  
            return rate * t

        elif drift_type == 'exponential':
            A = kwargs.get('A', 1.0)
            k = kwargs.get('k', 0.01)
            return A * (1 - np.exp(-k * t))

        elif drift_type == 'sinusoidal':
            A = kwargs.get('A', 0.2)
            f = kwargs.get('f', 1/300)  
            return A * np.sin(2 * np.pi * f * t)

        else:
            raise ValueError(f"Unknown drift type: {drift_type}")

    def clipping(self, signal, **kwargs):
        max_val = kwargs.get('max_val', self.amplitude)
        min_val = kwargs.get('min_val', -self.amplitude)
        return np.clip(signal, min_val, max_val)

    def non_linear_response(self, signal, response_type='tanh', **kwargs):
        """
        Applies a non-linear transformation to the signal to simulate sensor non-linearity.

        :param signal: Input signal (1D numpy array)
        :param response_type: Type of non-linearity ('tanh', 'sigmoid', 'polynomial')
        :param kwargs: Parameters depending on response_type
        :return: Non-linearly transformed signal
        """
        signal = np.asarray(signal)

        if response_type == 'tanh':
            scale = kwargs.get('scale', 1.0)
            return np.tanh(scale * signal)

        elif response_type == 'sigmoid':
            scale = kwargs.get('scale', 1.0)
            return 1 / (1 + np.exp(-scale * signal))

        elif response_type == 'polynomial':
            a = kwargs.get('a', 0.01)
            b = kwargs.get('b', 0.0)
            c = kwargs.get('c', 1.0)
            d = kwargs.get('d', 0.0)
            return a * signal**3 + b * signal**2 + c * signal + d

        else:
            raise ValueError(f"Unknown response_type: {response_type}")

    def motion_artifacts(self, value, artifact_type='spike', **kwargs):
        """
        Simulates motion artifacts caused by sensor movement or vibration.

        :param value (float): Clean signal value.
        :param artifact_type (str): Type of artifact ('spike', 'random_amplitude').
        :param kwargs: Parameters for artifact (e.g., prob, spike_amplitude).
        :return: Distorted signal value.
        """
        if artifact_type == 'spike':
            prob = kwargs.get('prob', 0.01)
            spike_amplitude = kwargs.get('spike_amplitude', self.amplitude * 2)
            if np.random.rand() < prob:
                direction = np.random.choice([-1, 1])
                return value + direction * spike_amplitude
            return value

        elif artifact_type == 'random_amplitude':
            prob = kwargs.get('prob', 0.01)
            min_val = kwargs.get('min_val', -self.amplitude)
            max_val = kwargs.get('max_val', self.amplitude)
            if np.random.rand() < prob:
                return np.random.uniform(min_val, max_val)
            return value

        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")

    def non_stationary_drift(self, t, method='random_walk', **kwargs):
        """
        Simulates non-stationary baseline drift over time.

        :param t: Time vector (1D numpy array)
        :param method: Method for generating drift ('random_walk', 'variable_sin', 'mixed')
        :param kwargs: Parameters like scale, frequency, etc.
        :return: Drift signal (same shape as t)
        """
        t = np.asarray(t)

        if method == 'random_walk':
            scale = kwargs.get('scale', 0.01)
            steps = np.random.normal(loc=0, scale=scale, size=len(t))
            return np.cumsum(steps)

        elif method == 'variable_sin':
            # sinusoidal drift with variable frequency
            base_freq = kwargs.get('base_freq', 1 / 300)
            mod_strength = kwargs.get('mod_strength', 0.5)
            freq = base_freq * (1 + mod_strength * np.sin(0.1 * 2 * np.pi * t))
            return kwargs.get('amplitude', 0.2) * np.sin(2 * np.pi * freq * t)

        elif method == 'mixed':
            # Combination of Random Walk + Sinusoidal
            walk = self.non_stationary_drift(t, method='random_walk', scale=kwargs.get('scale', 0.01))
            sinus = self.non_stationary_drift(t, method='variable_sin', amplitude=kwargs.get('amplitude', 0.2))
            return walk + sinus

        else:
            raise ValueError(f"Unknown method: {method}")

    def low_pass_filtering(self, signal, **kwargs):
        """
        Applies a low-pass Butterworth filter to the signal.

        :param signal: Input signal (1D numpy array)
        :param cutoff: Cutoff frequency in Hz
        :param order: Order of the Butterworth filter
        :return: Filtered signal
        """
        cutoff = kwargs.get('cutoff', 5)
        order = kwargs.get('order', 4)
        fs = self.sampling_rate 
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def moving_average(self, signal, **kwargs):
        """
        Applies a moving average filter to smooth the signal.

        :param signal: input signal
        :param window_size: number of samples for the moving window
        :return: smoothed signal
        """
        window_size = kwargs.get('window_size', 5)
        if window_size < 1:
            return signal
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(signal, kernel, mode='same')
        return smoothed
    
    def exponential_smoothing(self, signal, **kwargs):
        """
        Applies exponential smoothing to the signal.

        :param signal: input signal
        :param alpha: smoothing factor (0 < alpha <= 1)
        :return: exponentially smoothed signal
        """
        alpha = kwargs.get('alpha', 0.1)
        if not (0 < alpha <= 1):
            return signal
        smoothed = np.zeros_like(signal)
        smoothed[0] = signal[0]
        for i in range(1, len(signal)):
            smoothed[i] = alpha * signal[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

