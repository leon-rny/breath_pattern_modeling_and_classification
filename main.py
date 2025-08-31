import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import numpy as np
import matplotlib.pyplot as plt

from breath_modeling.core.transmitter import Transmitter
from breath_modeling.core.channel import Channel
from breath_modeling.core.receiver import Receiver

if __name__ == "__main__":
    # time vector
    fs = 1000  
    duration = 30 
    t = np.linspace(0, duration, int(fs * duration))

    # create transmitter
    tx = Transmitter()

    # generate signal
    func = tx.eupnea  # choose from: eupnea, tachypnea, kussmaul, cheyne_stokes, sighing, apnea
    signal = func(t)
    signal_name = func.__name__.capitalize().replace('Cheyne_stokes', 'Cheyne-Stokes')  

    # generate molecules from signal
    molecules = tx.molecular_output(signal, scale=1)

    # create channel
    ch = Channel()

    # create channel impulse response
    M = 1       # number of molecules
    D = 0.18    # diffusion coefficient
    r = 20      # distance in cm
    v = 30      # velocity in cm/s
    cir = ch.cir(t, M=M, D=D, r=r, v=v)

    # convolve molecules with channel impulse response
    convolved_signal = np.convolve(molecules, cir, mode='full')[:len(t)] * 1 / fs

    # add noise to the convolved signal
    noise_floor = 0.001     # noise floor
    base_strength = 0.001   # amplitude-dependent noise strength
    noisy_signal = ch.noise_floor(convolved_signal, noise_floor=noise_floor)
    noisy_signal = ch.amplitude_dependent_noise(noisy_signal, distance=r, base_strength=base_strength)

    # create receiver 
    rx = Receiver(model_type="rf")

    # predict signal using random forest model
    prediction_rf = rx.predict_signal_rf(noisy_signal, fs=fs)
    prediction_rf = prediction_rf.capitalize().replace('Cheyne_stokes', 'Cheyne-Stokes')

    # plot pattern with predictions
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(t, signal, label=f'{signal_name} Signal', color='tab:blue')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid()
    axs[0].legend(loc='upper right')

    axs[1].plot(t, convolved_signal, label='Convolved Signal', color='tab:orange')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid()
    axs[1].legend(loc='upper right')

    axs[2].plot(t, noisy_signal, label='Noisy Signal', color='tab:green')
    axs[2].set_ylabel('Amplitude')
    axs[2].set_title(f'Prediction: {prediction_rf}')
    axs[2].grid()
    axs[2].legend(loc='upper right')
    axs[2].set_xlabel('Time in s')
    plt.tight_layout()
    plt.show()

    # load sequences
    sequences = np.load(os.getcwd() + "/src/breath_modeling/model/dataset/sequences/sequence_000.npy")

    fs = 500
    window_size = 3.0
    stride = 1.5

    # create receiver for sequence prediction
    rx = Receiver(model_type="bilstm") # "cnn" or "bilstm"
    pred_labels, starts = rx.predict_window(sequences, fs=fs, window_size=window_size, stride=stride)

    # plot sequences with predictions
    t = np.arange(len(sequences)) / fs
    label_colors = {l.capitalize().replace('Cheyne_stokes', 'Cheyne-Stokes'): plt.cm.tab10(i) for i, l in enumerate(rx.label_encoder.classes_)}
    pred_labels_cap = [l.capitalize().replace('Cheyne_stokes', 'Cheyne-Stokes') for l in pred_labels]

    plt.figure(figsize=(14, 4))
    plt.plot(t, sequences, label="Signal", linewidth=1)
    for i, start in enumerate(starts):
        plt.axvspan(start, start + window_size, color=label_colors[pred_labels_cap[i]], alpha=0.3)

    plt.xlabel("Time in s")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=c, label=l, alpha=0.3) for l, c in label_colors.items()
    ], loc="upper right")
    plt.tight_layout()
    plt.show()