"""This file defines all relevant paths for dataset storage and access. By using relative paths based on the current file’s location,
it ensures portability and consistency across different environments."""
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(THIS_DIR, "dataset")
SAMPLES_DIR = os.path.join(DATASET_DIR, "samples")
SEQUENCE_SIGNAL_DIR = os.path.join(DATASET_DIR, "sequences")
SAMPLE_METADATA_FILE = os.path.join(SAMPLES_DIR, "samples_labels.jsonl")
WINDOW_LABELS_FILE = os.path.join(SEQUENCE_SIGNAL_DIR, "window_labels.jsonl")
SEQUENCE_SEGMENTS_FILE = os.path.join(SEQUENCE_SIGNAL_DIR, "sequences_labels.jsonl")
SEQUENCE_WINDOW_LABELS_FILE = os.path.join(SAMPLES_DIR, "sequence_windows_labels.jsonl")