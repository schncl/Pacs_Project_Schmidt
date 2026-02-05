# Multi-Target Classification Framework

A Python framework for multi-target classification using Neural Networks and Gaussian Processes, with MATLAB integration support.

## Overview

This framework provides tools for training and deploying multi-target classifiers on MATLAB datasets. It supports two model types:
- **Neural Networks** (PyTorch-based) - Fast, scalable, good for larger datasets
- **Gaussian Processes** (scikit-learn) - Probabilistic, uncertainty quantification, better for smaller datasets

## Project Structure

```
project/
├── src/
│   ├── modelbase.py          # Abstract base class for models
│   ├── modelnn.py             # Neural Network classifier
│   ├── modelGPC.py            # Gaussian Process classifier
│   ├── dataset.py             # MATLAB dataset loader
│   ├── config.py              # Configuration loader
│   └── matlab_interface.py    # MATLAB-Python bridge
├── configs/                   # Configuration files
├── models/                    # Saved trained models
├── data/                      # MATLAB dataset files
└── README.md
```

## Installation

### Requirements

```bash
pip install -r requirement.txt
```

## Usage

### 1. Prepare Your Configuration

Create `configs/experiment_name.json`:

```json
{
    "dataset_path": "../data/path_to_dataset",
    "input_fields": ["name_of_feature1","name_of_feature2",...],
    "target_fields": ["name_of_target1","name_of_target2",...],
    
    "model":"both", 
    "samples_gp":100,
    "plots":true,
    "save_model":true,
    "path_to_model_nn":"../models/model_heat_nn.pt",
    "path_to_model_gp": "../models/model_heat_gp.joblib",


    "neural_network": {
        "epochs": 5000,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_layers": [64, 128],
        "dropout_rate": 0.2,
        "verbose": true
      },
    

    "gaussian_process": {
        "n_restarts_optimizer": 10,
        "random_state": 42,
        "constant_value": 1.0,
        "constant_bounds": [1e-3, 1e3],
        "rbf_length_scale": 1.0,
        "rbf_bounds": [1e-2, 1e2]
      }
    
}
```

### 2. Load Your Dataset

```python
from dataset import MatDataset
from config import load_config

# Load configuration
config = load_config("configs/your_config.json")

# Load MATLAB dataset
dataset = MatDataset("data/your_dataset.mat")
dataset.load()

# Prepare ML data
X, Y = dataset.prepare_ml_data(
    input_fields=config["dataset"]["input_fields"],
    target_fields=config["dataset"]["target_fields"]
)
```

### 3. Train a Neural Network

```python
from modelnn import ModelNN

# Initialize model
model = ModelNN(config["neural_network"])

# Train
X_test, Y_test = model.train_model(X, Y)

# Evaluate
model.compute_metrics(X_test, Y_test, target_names=["Target 1", "Target 2"])
model.plot_confusions(X_test, Y_test, target_names=["Target 1", "Target 2"])

# Save
model.save("models/my_nn_model.pth")
```

### 4. Train a Gaussian Process

```python
from modelGPC import GPModel

# Initialize model
model = GPModel(config["gaussian_process"])

# Train (uses subset of data if specified in config)
X_test, Y_test = model.train_model(X, Y)

# Evaluate
model.compute_metrics(X_test, Y_test, target_names=["Target 1", "Target 2"])
model.plot_confusions(X_test, Y_test, target_names=["Target 1", "Target 2"])

# Save
model.save("models/my_gp_model.pkl")
```

### 5. Make Predictions

```python
# Load trained model
from modelnn import ModelNN

model = ModelNN(config["neural_network"])
model.load("models/my_nn_model.pth")

# Predict on new data
X_new = np.array([[1.0, 2.0]])
predictions = model.predict(X_new)
# Returns: {"target_0": [predicted_class], "target_1": [predicted_class]}
```

## MATLAB Integration

### Setup MATLAB Python Environment

In MATLAB:
```matlab

% set Python executable
pyenv('Version', '/path/to/python')
```

### Load and Use Models from MATLAB

```matlab
% Add Python module path
folder='path_to_src';
if count(py.sys.path, folder) == 0
    insert(py.sys.path, int32(0), folder);
end

% Load Neural Network model
model = py.matlab_interface.load_model_nn('my_nn_model.pth', 'your_config.json');

% Make prediction
X = [1.0, 2.0];  % Your input features
result = py.matlab_interface.predict_nn(model, X);

% Access predictions
target_0_pred = result{'target_0'};
target_1_pred = result{'target_1'};
```

### Load GP Model from MATLAB

```matlab
% Load Gaussian Process model
gp_model = py.matlab_interface.load_gp_model('my_gp_model.pkl', 'config.json');

% Predict
result = py.matlab_interface.gp_predict(gp_model, X);
```

## Dataset Format

Your MATLAB `.mat` file (v7.3) should have this structure:

```matlab
% In MATLAB, create dataset
dataset.feature1 = [1.0; 2.0; 3.0; ...];
dataset.feature2 = [4.0; 5.0; 6.0; ...];
dataset.feature3 = [7.0; 8.0; 9.0; ...];
dataset.target1 = [0; 1; 0; ...];      % Categorical labels
dataset.target2 = [1; 1; 0; ...];      % Categorical labels

% Save as v7.3
save('my_dataset.mat', 'dataset', '-v7.3');
```

**Important:** 
- Use `-v7.3` flag when saving (HDF5 format)
- Top-level structure must be named `dataset`
- Features can be continuous values
- Targets should be categorical labels

## Configuration Guide

### Neural Network Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `epochs` | Training iterations | 300 | 50-5000 |
| `batch_size` | Samples per batch | 32 | 8-256 |
| `learning_rate` | Optimizer step size | 0.001 | 1e-5 to 0.1 |
| `hidden_layers` | Network architecture | [64, 32] | Any list |
| `dropout_rate` | Regularization | 0.3 | 0.0-0.5 |
| `optimizer` | Optimization algorithm | "adam" | "adam", "sgd" |
| `verbose` | Print training progress | false | true/false |

### Gaussian Process Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `n_restarts_optimizer` | Kernel hyperparameter restarts | 10 | 5-20 |
| `random_state` | Reproducibility seed | 42 | Any int |
| `constant_value` | Initial kernel constant | 1.0 | 0.1-10 |
| `rbf_length_scale` | Initial RBF scale | 1.0 | 0.1-10 |

### General Parameters
| Parameter | Description |
|-----------|-------------|
| `dataset_path` | path to dataset |
| `input_fields` | Names of input fields 
| `target_fields` | Names of target fields 
| `model` | Which model to train, can be "nn","gp","both" 
| `samples_gp` | How many samples to use for gp training, should be <100
| `plot` | Activate visualization |
| `save_model` | If the model is saved |
| `path_to_model_nn` | Where the nn model will be saved |
| `path_to_model_gp` | Where the gp model will be saved |



**Note:** Gaussian Processes are O(n³) in training time. Use `samples_gp` to limit dataset size.

