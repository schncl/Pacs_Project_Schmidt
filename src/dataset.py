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