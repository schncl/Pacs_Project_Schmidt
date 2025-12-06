import os
from typing import Dict, Any, List, Tuple

import numpy as np
import mat73


class MatDataset:
    """Class that loads a Matlab 7.3 dataset and converts it to numpy arrays"""


    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        self.filepath = filepath
        self.data: Dict[str, Any] = {}
        self._loaded = False



    def load(self) -> None:
        """Load MATLAB dataset into self.data."""
        data_dict = mat73.loadmat(self.filepath)

        key = next((k for k in data_dict if k.lower() == "dataset"), None)
        if key is None:
            raise ValueError(f"No 'dataset' key found. Keys: {list(data_dict.keys())}")

        raw = data_dict[key]

        # This step is needed as Matlab has no scalar
        cleaned = {
            k: [v.item() if isinstance(v, np.ndarray) and v.size == 1 else v for v in vals] for k, vals in raw.items()
        }

        self.data = cleaned
        self._loaded = True
        

    def validate(self) -> None:
        """Check whether the dataset is loaded and non-empty."""
        if not self._loaded:
            raise RuntimeError("Dataset not loaded")
        if not self.data:
            raise ValueError("Dataset is empty")



    @staticmethod
    def _to_column(arr):
        """Ensure arr has shape (N, 1)."""
        arr = np.asarray(arr)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    def prepare_ml_data(
        self,
        input_fields: List[str],
        target_fields: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert raw data into numpy arrays.

        Returns:
            X: shape (N, n_input_features)
            Y: shape (N, n_target_features)
        """
        self.validate()

        X = np.hstack([self._to_column(self.data[f]) for f in input_fields])
        Y = np.hstack([self._to_column(self.data[f]) for f in target_fields])

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Inputs and targets have different number of rows")

        return X, Y



