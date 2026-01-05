## @file matlab_interface.py
#  @brief MATLAB interface for loading and using trained ML models
#
#  This module provides a Python-MATLAB bridge for loading trained
#  Neural Network and Gaussian Process models and making predictions.
#  Designed to be called from MATLAB using Python engine
#
#
#  @warning Global variables are used to maintain model state across calls


import os
from modelnn import ModelNN
from modelGPC import GPModel
from config import load_config


## @var current_dir
#  Directory containing this matlab_interface.py file
current_dir = os.path.dirname(os.path.abspath(__file__))

## @var project_root
#  Root directory of the project (parent of current_dir)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

## @var CONFIG_PATH
#  Full path to the configuration JSON file
#  Expected location: <project_root>/configs/config.json
# @note The models are reconstructed starting from the provided json file
CONFIG_PATH = os.path.join(project_root, "configs", "config.json")

## @var MODELS_DIR
#  Directory containing saved model files
#  Expected location: <project_root>/models/
MODELS_DIR = os.path.join(project_root, "models")


## @brief Load a trained Neural Network model
#
#  Loads a previously trained and saved Neural Network model from memory
#  Initializes the model architecture using configuration from config.json
#
#  @param filepath Name of the model file 
#    Example: "model_family1.pt" or "experiment_v2.pt"
#
#  @return Loaded ModelNN instance ready for predictions
#
#  @note Model architecture is determined by config.json, not the saved file
def load_model_nn(filepath):
    """Load the model weights"""
    
    global global_model_nn
    
    config = load_config(CONFIG_PATH)

    model_path = os.path.join(MODELS_DIR, filepath)

    global_model_nn = ModelNN(config['neural_network'])

    global_model_nn.load(model_path)

    return global_model_nn


## @brief Make predictions using loaded Neural Network model
#
#  Uses a loaded Neural Network model to predict target values for
#  given input features. Handles automatic conversion from MATLAB
#  arrays or Python lists to the required numpy format.
#
#  @param model Loaded ModelNN instance (from load_model_nn)
#  @param X Input features
#
#  @return Dictionary mapping target names to predicted labels
#    Format: {"target_0": [predicted_label], "target_1": [...], ...}
def predict_nn(model, X):
    """Predict using the loaded model. X can be a list or numpy array."""
    import numpy as np

    X_np = np.array(X, dtype=float)
    
  
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)
    

    result = model.predict(X_np)
    
    return result


## @brief Load a trained Gaussian Process model
#
#  Loads a previously trained and saved Gaussian Process model from disk.
#  Initializes GP classifiers using configuration from config.json
#
#  @param filepath Name of the model file 
#
#  @return Loaded GPModel instance ready for predictions
def load_gp_model(filepath):
    """Load the GP model"""
    global global_model_gp

    config = load_config(CONFIG_PATH)

    model_path = os.path.join(MODELS_DIR, filepath)

    global_model_gp = GPModel(config['gaussian_process'])
    global_model_gp.load(model_path)

    return global_model_gp


## @brief Make predictions using loaded Gaussian Process model
#
#  Uses a loaded GP model to predict target values
#  @param model Loaded GPModel instance (from load_gp_model)
#  @return Dictionary mapping target names to predicted labels
#    Format: {"target_0": [predicted_label], "target_1": [...], ...}
def gp_predict(model, X):
    """Predict using loaded GPModel. X can be a list or numpy array."""
    import numpy as np
    

    X_np = np.array(X, dtype=float)
    
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)
    
    result = model.predict(X_np)
    
    return result