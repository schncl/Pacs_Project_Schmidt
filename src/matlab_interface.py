

import os
from modelnn import ModelNN
from modelGPC import GPModel
from config import load_config


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir,".."))

CONFIG_PATH = os.path.join(project_root, "configs", "config.json")
MODELS_DIR = os.path.join(project_root, "models")



def load_model_nn(filepath):
    """Load the model weights"""
    
    
    global global_model_nn
    config = load_config(CONFIG_PATH)
    model_path = os.path.join(MODELS_DIR, filepath)
    global_model_nn = ModelNN(config['neural_network'])
    global_model_nn.load(model_path)

    return global_model_nn


def predict_nn(model, X):
    """Predict using the loaded model. X can be a list or numpy array."""
    import numpy as np


    X_np = np.array(X, dtype=float)
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)
    

    result = model.predict(X_np)
    
    return result


def load_gp_model(filepath):
    "Load the gp model"
    global global_model_gp

    config = load_config(CONFIG_PATH)

    model_path = os.path.join(MODELS_DIR, filepath)

    global_model_gp = GPModel(config['gaussian_process'])
    global_model_gp.load(model_path)

    return global_model_gp


def gp_predict(model, X):
    """Predict using loaded GPModel. X can be a list or numpy array."""
    import numpy as np
    

    X_np = np.array(X, dtype=float)
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)
    

    result = model.predict(X_np)
    return result