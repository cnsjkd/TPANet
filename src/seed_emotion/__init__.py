"""
SEED emotion recognition utilities.

This package bundles data augmentation, preprocessing, and model training
components prepared for open-source distribution alongside the manuscript.
"""

from .data_augmentation import augment_data
from .chunk_preprocessing import preprocess_and_save_chunks_per_file

__all__ = [
    "augment_data",
    "preprocess_and_save_chunks_per_file",
]
