import os
import json
import time

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

from breath_modeling.core.transmitter import Transmitter
from breath_modeling.core.channel import Channel

from breath_modeling.model.config import (
    SAMPLES_DIR, SAMPLE_METADATA_FILE,
    SEQUENCE_SIGNAL_DIR, SEQUENCE_SEGMENTS_FILE,
    WINDOW_LABELS_FILE, SEQUENCE_WINDOW_LABELS_FILE
)

def generate_pattern(tx, ch, pattern, FS, pattern_durations,
                            TIME_VARIATION, AMPLITUDE_VARIATION, FREQUENCY_VARIATION,
                            NUM_MOLECULES, DIFFUSION_COEFFICIENT, DISTANCE, VELOCITY,
                            sequence, NOISE_FLOOR, BASE_STRENGTH):
    """    Generates a breathing pattern signal.
    :param tx: Transmitter object
    :param ch: Channel object
    :param pattern: Breathing pattern name
    :param FS: Sampling frequency
    :param pattern_durations: Dictionary of pattern durations
    :param TIME_VARIATION: Time variation factor
    :param AMPLITUDE_VARIATION: Amplitude variation factor
    :param FREQUENCY_VARIATION: Frequency variation factor
    :param NUM_MOLECULES: Number of molecules
    :param DIFFUSION_COEFFICIENT: Diffusion coefficient
    :param DISTANCE: Distance
    :param VELOCITY: Velocity
    :param sequence: Whether to generate a sequence or a single sample
    :param NOISE_FLOOR: Noise floor
    :param BASE_STRENGTH: Base strength
    :return: Generated signal, time vector, amplitude variation, frequency variation, noise floor, base strength
    """
    
    method = getattr(tx, pattern)
    
    # time and duration
    base_duration = pattern_durations[pattern]
    # normal distribution for duration
    duration = np.random.normal(loc=base_duration, scale=TIME_VARIATION * base_duration)
    t = np.linspace(0, duration, int(FS * duration))
    
    # create signal
    amp_var = np.random.uniform(1 - AMPLITUDE_VARIATION, 1 + AMPLITUDE_VARIATION)
    freq_var = np.random.uniform(1 - FREQUENCY_VARIATION, 1 + FREQUENCY_VARIATION)
    signal = method(t, amplitude_factor=amp_var, frequency_factor=freq_var)
    molecules = tx.molecular_output(signal, scale=NUM_MOLECULES)

    # apply cir
    cir = ch.cir(t, M=NUM_MOLECULES, D=DIFFUSION_COEFFICIENT, r=DISTANCE, v=VELOCITY)
    cir = cir / np.max(cir)
    cir_signal = fftconvolve(molecules, cir, mode='full')[:len(t)] * 1 / FS

    # noise
    if sequence:
        noise_floor = NOISE_FLOOR
        base_strength = BASE_STRENGTH
    else:
        noise_floor = abs(np.random.normal(loc=0.0, scale=NOISE_FLOOR))
        base_strength = abs(np.random.normal(loc=0.0, scale=BASE_STRENGTH))
    noisy = ch.noise_floor(cir_signal, noise_floor=noise_floor)
    noisy = ch.amplitude_dependent_noise(noisy, distance=DISTANCE, base_strength=base_strength)

    # attenuation
    final_signal = ch.attenuation(noisy, t, distance=DISTANCE)

    return final_signal, t, amp_var, freq_var, noise_floor, base_strength

def generate_sample_dataset(
    tx, ch, FS, pattern_all, pattern_durations, NUM_SAMPLES,
    SAMPLES_DIR, SAMPLES_LABELS,
    TIME_VARIATION, AMPLITUDE_VARIATION, FREQUENCY_VARIATION,
    NUM_MOLECULES, DIFFUSION_COEFFICIENT, DISTANCE, VELOCITY,
    NOISE_FLOOR, BASE_STRENGTH,
    plot=False
    ):
    """
    Generates a dataset of breathing patterns as samples.

    :param tx: Transmitter object
    :param ch: Channel object
    :param FS: Sampling frequency
    :param pattern_all: List of all patterns
    :param pattern_durations: Dictionary of pattern durations
    :param NUM_SAMPLES: Number of samples to generate
    :param SAMPLES_DIR: Directory to save samples
    :param SAMPLES_LABELS: File to save sample labels
    :param TIME_VARIATION: Time variation factor
    :param AMPLITUDE_VARIATION: Amplitude variation factor
    :param FREQUENCY_VARIATION: Frequency variation factor
    :param NUM_MOLECULES: Number of molecules
    :param DIFFUSION_COEFFICIENT: Diffusion coefficient
    :param DISTANCE: Distance
    :param VELOCITY: Velocity
    :param NOISE_FLOOR: Noise floor
    :param BASE_STRENGTH: Base strength
    :param plot: Whether to plot samples
    """
    # create directory for samples
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    if os.path.exists(SAMPLES_LABELS):
        os.remove(SAMPLES_LABELS)

    meta_data_list = []

    # loop through all patterns and generate samples
    for pattern in pattern_all:
        for i in range(NUM_SAMPLES):
            signal, t, amp_var, freq_var, noise_floor, base_strength = generate_pattern(
                tx, ch, pattern, FS, pattern_durations,
                TIME_VARIATION, AMPLITUDE_VARIATION, FREQUENCY_VARIATION,
                NUM_MOLECULES, DIFFUSION_COEFFICIENT, DISTANCE, VELOCITY,
                sequence=False,
                NOISE_FLOOR=NOISE_FLOOR,
                BASE_STRENGTH=BASE_STRENGTH
            )

            # create metadata
            meta_data_list.append({
                "pattern": pattern,
                "amplitude_variation": amp_var,
                "frequency_variation": freq_var,
                "duration": t[-1],
                "duration_variation": TIME_VARIATION,
                "num_molecules": NUM_MOLECULES,
                "distance": DISTANCE,
                "diffusion_coefficient": DIFFUSION_COEFFICIENT,
                "velocity": VELOCITY,
                "noise_floor": noise_floor,
                "base_strength": base_strength,
                "sample_index": i
            })

            output_file = os.path.join(SAMPLES_DIR, f"{pattern}_{i}.npy")
            np.save(output_file, signal)

            if plot and i % NUM_SAMPLES == 0:
                plt.plot(t, signal)
                plt.title(f"{pattern} Sample {i}")
                plt.grid()
                plt.xlabel("Time [s]")
                plt.ylabel("Amplitude")
                plt.tight_layout()
                plt.show()

    # Write all metadata as JSONL
    with open(SAMPLES_LABELS, 'w') as f:
        for meta_data in meta_data_list:
            f.write(json.dumps(meta_data) + '\n')

def generate_sequence_dataset(
    tx, ch, FS, pattern_all, pattern_durations,
    SEQUENCE_DURATION, SAMPLES_PER_SEQUENCE,
    SEQUENCE_DIR, SEQUENCE_LABELS, WINDOW_LABELS, SEQUENCES_WINDOWS_LABELS,
    TIME_VARIATION, AMPLITUDE_VARIATION, FREQUENCY_VARIATION,
    NUM_MOLECULES, DIFFUSION_COEFFICIENT, DISTANCE, VELOCITY,
    NOISE_FLOOR, BASE_STRENGTH, plot=False,
    enable_window_output=True, window_size=3.0, window_stride=1.5
    ):
    """
    Generates a dataset of breathing patterns as sequences
    :param tx: Transmitter object
    :param ch: Channel object
    :param FS: Sampling frequency
    :param pattern_all: List of all patterns
    :param pattern_durations: Dictionary of pattern durations
    :param SEQUENCE_DURATION: Duration of each sequence in seconds
    :param SAMPLES_PER_SEQUENCE: Number of sequences to generate
    :param SEQUENCE_DIR: Directory to save sequences
    :param SEQUENCE_LABELS: File to save sequence labels
    :param WINDOW_LABELS: File to save window labels
    :param SEQUENCES_WINDOWS_LABELS: File to save sequence windows labels
    :param TIME_VARIATION: Time variation factor
    :param AMPLITUDE_VARIATION: Amplitude variation factor
    :param FREQUENCY_VARIATION: Frequency variation factor
    :param NUM_MOLECULES: Number of molecules
    :param DIFFUSION_COEFFICIENT: Diffusion coefficient
    :param DISTANCE: Distance
    :param VELOCITY: Velocity
    :param NOISE_FLOOR: Noise floor
    :param BASE_STRENGTH: Base strength
    :param plot: Whether to plot sequences
    :param enable_window_output: Whether to enable sliding window output
    :param window_size: Size of the sliding window in seconds
    :param window_stride: Stride of the sliding window in seconds
    """
    # create directory for samples
    os.makedirs(SEQUENCE_DIR, exist_ok=True)

    # delete old label files if they exist
    for path in [SEQUENCE_LABELS, WINDOW_LABELS, SEQUENCES_WINDOWS_LABELS]:
        if os.path.exists(path):
            os.remove(path)

    # open files for window output
    if enable_window_output:
        window_fh = open(WINDOW_LABELS, "a")
        sequence_sets_fh = open(SEQUENCES_WINDOWS_LABELS, "a")

    # Prepare breathing patterns
    min_classes_per_sequence = 2
    n_total_patterns_needed = SAMPLES_PER_SEQUENCE * min_classes_per_sequence
    pattern_pool = (pattern_all * ((n_total_patterns_needed // len(pattern_all)) + 1))[:n_total_patterns_needed]
    np.random.shuffle(pattern_pool)

    for seq_idx in range(SAMPLES_PER_SEQUENCE):
        combined_signal = np.zeros(int(SEQUENCE_DURATION * FS))
        segment_labels = []
        sequence_windows = []
        sequence_labels = []
        current_index = 0

        # random number of patterns per sequence
        n_patterns = min_classes_per_sequence + np.random.randint(0, 3)
        if len(pattern_pool) < n_patterns:
            pattern_pool += pattern_all * ((n_patterns // len(pattern_all)) + 1)
            np.random.shuffle(pattern_pool)
        patterns_for_this_sequence = [pattern_pool.pop() for _ in range(n_patterns)]
        np.random.shuffle(patterns_for_this_sequence)

        # random noise level
        pattern_idx = 0
        noise_floor = abs(np.random.normal(loc=0.0, scale=NOISE_FLOOR))
        base_strength = abs(np.random.normal(loc=0.0, scale=BASE_STRENGTH))

        # build signal
        while current_index < len(combined_signal):
            pattern = patterns_for_this_sequence[pattern_idx % len(patterns_for_this_sequence)]
            pattern_idx += 1

            signal, t, _, _, _, _ = generate_pattern(
                tx, ch, pattern, FS, pattern_durations,
                TIME_VARIATION, AMPLITUDE_VARIATION, FREQUENCY_VARIATION,
                NUM_MOLECULES, DIFFUSION_COEFFICIENT, DISTANCE, VELOCITY,
                sequence=True, NOISE_FLOOR=noise_floor, BASE_STRENGTH=base_strength)

            remaining = len(combined_signal) - current_index
            if len(signal) > remaining:
                signal = signal[:remaining]
                t = t[:remaining]

            end_index = current_index + len(signal)
            combined_signal[current_index:end_index] = signal

            segment_labels.append({
                "pattern": pattern,
                "start_index": current_index,
                "end_index": end_index,
                "start_time": current_index / FS,
                "end_time": end_index / FS,
                "sequence_index": seq_idx
            })

            current_index = end_index

        # Save the complete sequence
        np.save(os.path.join(SEQUENCE_DIR, f"sequence_{seq_idx:03}.npy"), combined_signal)

        # Write segment labels
        with open(SEQUENCE_LABELS, "a") as f:
            f.write(json.dumps(segment_labels) + "\n")

        # Visualization
        if plot:
            color_map = plt.get_cmap("tab10")
            t_combined = np.linspace(0, SEQUENCE_DURATION, len(combined_signal))
            plt.plot(t_combined, combined_signal)
            for seg in segment_labels:
                plt.axvspan(seg["start_time"], seg["end_time"],
                            alpha=0.3, label=seg["pattern"],
                            color=color_map(pattern_all.index(seg["pattern"]) % 10))
            plt.legend(loc='upper right')
            plt.title(f"Sequence {seq_idx}")
            plt.grid()
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.show()

        # Sliding Windows & Sequence Grouping
        if enable_window_output:
            win_len = int(window_size * FS)
            stride_len = int(window_stride * FS)
            num_windows = (len(combined_signal) - win_len) // stride_len + 1

            for w_idx in range(num_windows):
                start_idx = w_idx * stride_len
                end_idx = start_idx + win_len
                t_start = start_idx / FS
                t_end = end_idx / FS
                window = combined_signal[start_idx:end_idx]

                overlapping = [
                    seg["pattern"]
                    for seg in segment_labels
                    if not (seg["end_time"] <= t_start or seg["start_time"] >= t_end)
                ]
                if not overlapping:
                    continue

                label = max(set(overlapping), key=overlapping.count)

                # CNN-Saving
                window_record = {
                    "sequence_index": seq_idx,
                    "window_index": w_idx,
                    "start_time": t_start,
                    "end_time": t_end,
                    "label": label,
                    "signal": window.tolist()
                }
                window_fh.write(json.dumps(window_record) + "\n")

                # LSTM-Saving
                sequence_windows.append(window.tolist())
                sequence_labels.append(label)

            sequence_record = {
                "sequence_index": seq_idx,
                "windows": sequence_windows,
                "labels": sequence_labels
            }
            sequence_sets_fh.write(json.dumps(sequence_record) + "\n")

    if enable_window_output:
        window_fh.close()
        sequence_sets_fh.close()

if __name__ == "__main__":
    # start time
    start_time = time.time()

    # parameters
    FS = 500                            # Hz
    DISTANCE = 20                       # cm
    VELOCITY = 30                       # cm/s
    DIFFUSION_COEFFICIENT = 0.18        # cm^2/s
    NUM_MOLECULES = 1

    TIME_VARIATION = 0.25               # %
    AMPLITUDE_VARIATION = 0.1           # %
    FREQUENCY_VARIATION = 0.1           # %

    NOISE_FLOOR = 0.001                 # noise floor: 0.001
    BASE_STRENGTH = 0.001               # base strength: 0.0025

    SEQUENCE_DURATION = 60
    SAMPLES_PER_SEQUENCE = 500
    WINDOW_SIZE = 3.0                   # seconds maybe change the size
    WINDOW_STRIDE = 1.5                   # seconds

    NUM_SAMPLES = 500
    plot = False

    # objects
    tx = Transmitter()
    ch = Channel()

    # breathing patterns and durations
    pattern_all = ['eupnea', 'tachypnea', 'kussmaul', 'cheyne_stokes', 'sighing', 'apnea']

    pattern_durations = {'eupnea': 30, 'tachypnea': 30, 'kussmaul': 30, 'cheyne_stokes': 30, 'sighing': 30, 'apnea': 30}

    # generate datasets
    generate_sample_dataset(
        tx, ch, FS, pattern_all, pattern_durations, NUM_SAMPLES,
        SAMPLES_DIR, SAMPLE_METADATA_FILE,
        TIME_VARIATION, AMPLITUDE_VARIATION, FREQUENCY_VARIATION,
        NUM_MOLECULES, DIFFUSION_COEFFICIENT, DISTANCE, VELOCITY,
        NOISE_FLOOR, BASE_STRENGTH,
        plot=plot
    )

    pattern_durations = {'eupnea': 20, 'tachypnea': 5, 'kussmaul': 8, 'cheyne_stokes': 30, 'sighing': 4, 'apnea': 10}

    # generate sequence dataset
    generate_sequence_dataset(
        tx, ch, FS, pattern_all, pattern_durations,
        SEQUENCE_DURATION, SAMPLES_PER_SEQUENCE,
        SEQUENCE_SIGNAL_DIR, SEQUENCE_SEGMENTS_FILE,
        WINDOW_LABELS_FILE, SEQUENCE_WINDOW_LABELS_FILE,
        TIME_VARIATION, AMPLITUDE_VARIATION, FREQUENCY_VARIATION,
        NUM_MOLECULES, DIFFUSION_COEFFICIENT, DISTANCE, VELOCITY,
        NOISE_FLOOR, BASE_STRENGTH,
        plot=plot, enable_window_output=True,
        window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE
    )

    print(f"Done in {time.time() - start_time:.2f} seconds.")