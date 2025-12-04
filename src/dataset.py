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