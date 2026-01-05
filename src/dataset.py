## @file dataset.py
#  @brief MATLAB dataset loader and preprocessor for machine learning
#
#  This module provides functionality to load MATLAB v7.3 (.mat) files
#  and convert them into numpy arrays 

import os
from typing import Dict, Any, List, Tuple

import numpy as np
import mat73

## @class MatDataset
#  @brief Loader for MATLAB 7.3 format datasets
#
#  This class handles loading and preprocessing of MATLAB v7.3 files,
#  which are HDF5-based and require special handling. It provides
#  methods to:
#  - Load .mat files into Python dictionaries
#  - Validate data integrity
#  - Convert to numpy arrays for ML pipelines
#  - Handle MATLAB's scalar representation quirks
#
#  @note The class expects MATLAB files with a top-level 'dataset' structure
#  containing named fields for features and targets.
class MatDataset:
    """Class that loads a Matlab 7.3 dataset and converts it to numpy arrays"""

    ## @brief Constructor for MatDataset
    #
    #  Initializes the dataset loader with a file path.
    #  Validates that the file exists but does not load it yet.
    #
    #  @param filepath Path to the MATLAB .mat file (must be v7.3 format)
    #  @throws FileNotFoundError If the specified file does not exist
    #
    #  @note Call load() after construction to actually read the file
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        ## @var filepath
        #  Path to the MATLAB dataset file
        self.filepath = filepath
        ## @var data
        #  Dictionary storing loaded MATLAB data
        self.data: Dict[str, Any] = {}

        ## @var _loaded
        #  Internal flag indicating whether data has been loaded
        self._loaded = False


    ## @brief Load MATLAB dataset from file
    #
    #  Reads the MATLAB v7.3 file and extracts data into self.data.
    #  Performs the following steps:
    #   Load .mat file using mat73
    #   Find 'dataset' key (case-insensitive)
    #   Clean MATLAB scalar
    #   Store in self.data dictionary
    #   Set loaded flag
    #
    #  @throws ValueError If no 'dataset' key found in the MATLAB file
    #
    #  @note MATLAB stores scalars as 1-element arrays; this method unwraps them
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
        
    ## @brief Validate that dataset is loaded and not empty
    #
    #  Checks internal state to ensure:
    #  - Data has been loaded via load() method
    #  - Loaded data is not empty
    #
    #  @throws RuntimeError If load() has not been called yet
    #  @throws ValueError If the dataset is empty
    def validate(self) -> None:
        """Check whether the dataset is loaded and non-empty."""
        if not self._loaded:
            raise RuntimeError("Dataset not loaded")
        if not self.data:
            raise ValueError("Dataset is empty")



    ## @brief Convert array to column vector format
    #
    #  Ensures an array has shape (N, 1) for consistent stacking.
    #  Handles both 1D and multi-dimensional arrays.
    #
    #  @param arr Input array (any shape)
    #  @return Array reshaped to (N, 1) if 1D, otherwise unchanged
    @staticmethod
    def _to_column(arr):
        """Ensure arr has shape (N, 1)."""
        arr = np.asarray(arr)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr
    


    ## @brief Prepare machine learning data arrays
    #
    #  Extracts specified fields from loaded MATLAB data and converts
    #  them to properly shaped numpy arrays for ML training.
    #
    #  Perform the following steps:
    #   Validate that data is loaded
    #   Extract and stack input fields into X matrix
    #   Extract and stack target fields into Y matrix
    #   Verify dimensions match
    #
    #  @param input_fields List of field names to use as input features
    #  @param target_fields List of field names to use as target variables
    #
    #  @return Tuple (X, Y) where:
    #    - X: Input features array of shape (n_samples, n_input_features)
    #    - Y: Target labels array of shape (n_samples, n_target_features)
    #  @throws ValueError If X and Y have different number of rows
    def prepare_ml_data(
        self,
        input_fields: List[str],
        target_fields: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert raw data into numpy arrays.
        """
        self.validate()

        X = np.hstack([self._to_column(self.data[f]) for f in input_fields])
        Y = np.hstack([self._to_column(self.data[f]) for f in target_fields])

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Inputs and targets have different number of rows")

        return X, Y



