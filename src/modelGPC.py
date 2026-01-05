## @file modelGPC.py
#  @brief Gaussian Process classifier for multi-target classification
#
#  This module implements a Gaussian Process-based classifier that can
#  handle multiple target variables simultaneously. Each target is modeled
#  by a separate Gaussian Process classifier with RBF kernel.

from modelbase import BaseModel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from joblib import dump,load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


## @class GPModel
#  @brief Gaussian Process multi-target classifier
#
#  Implements a Gaussian Process classifier for multi-target classification
#  problems. Uses RBF (Radial Basis Function) kernel with constant scaling
#  for each target variable.
class GPModel(BaseModel):
    """Gaussian Process multi-target classifier"""



    ## @brief Constructor for GPModel  
    #   
    #  Initializes the Gaussian Process model with specified hyperparameters.
    #
    #  @param config Dictionary containing GP hyperparameters:
    #    - n_restarts_optimizer: Number of optimizer restarts (default: 10)
    #    - random_state: Random seed for reproducibility (default: 42)
    #    - constant_value: Initial constant kernel value (default: 1.0)
    #    - constant_bounds: Bounds for constant kernel [min, max] (default: [1e-3, 1e3])
    #    - rbf_length_scale: Initial RBF length scale (default: 1.0)
    #    - rbf_bounds: Bounds for RBF length scale [min, max] (default: [1e-2, 1e2])
    #  @param model_name String identifier for this model (default: "GPModel")

    def __init__(self, config, model_name="GPModel"):

        super().__init__(model_name="ModelNN")
        ## @var n_restarts_optimizer
        #  Number of times the optimizer is restarted to find global optimum
        self.n_restarts_optimizer = config.get("n_restarts_optimizer", 10)

        ## @var random_state
        #  Random seed for reproducibility of results
        self.random_state = config.get("random_state", 42)

        ## @var constant_value
        #  Initial value for the constant kernel multiplier
        self.constant_value = config.get("constant_value", 1.0)

        ## @var constant_bounds
        #  Optimization bounds for constant kernel parameter (min, max)
        self.constant_bounds = tuple(config.get("constant_bounds", [1e-3, 1e3]))

        ## @var rbf_length_scale
        #  Initial length scale for RBF kernel 
        self.rbf_length_scale = config.get("rbf_length_scale", 1.0)

        ## @var rbf_bounds
        #  Optimization bounds for RBF length scale (min, max)
        self.rbf_bounds = tuple(config.get("rbf_bounds", [1e-2, 1e2]))

        ## @var models
        #  Dictionary storing separate GP classifier for each target
        #  Key: target index, Value: trained GaussianProcessClassifier
        self.models = {}

        ## @var scaler
        #  StandardScaler for feature normalization
        self.scaler = None

        ## @var label_mappings
        #  Dictionary mapping original labels to encoded integers for each target
        #  Key: target index, Value: {original_label: encoded_int}
        self.label_mappings = {}

        ## @var input_min
        #  Minimum values of input features (for visualization bounds)
        self.input_min = None

        ## @var input_max
        #  Maximum values of input features (for visualization bounds)
        self.input_max = None



    ## @brief Preprocess data for Gaussian Process training
    #
    #  Performs the following steps:
    #  1. Splits data into 80% training, 20% testing
    #  2. Scales features using StandardScaler (fit on training only)
    #  3. Encodes categorical labels to integers for each target
    #  4. Stores min/max values for visualization
    #
    #  @param X Input features array of shape (n_samples, n_features)
    #  @param Y Target labels array of shape (n_samples, n_targets)
    #  @return Tuple (X_train, Y_train_encoded, X_test, Y_test)
    #    - X_train: Scaled training features (n_train_samples, n_features)
    #    - Y_train_encoded: Encoded training labels (n_train_samples, n_targets)
    #    - X_test: Scaled test features (n_test_samples, n_features)
    #    - Y_test: Original test labels (n_test_samples, n_targets)
    def preprocess_data(self, X, Y):

        """Apply preprocessing to Data"""
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]

        self.input_min = X.min(axis=0)
        self.input_max = X.max(axis=0)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.label_mappings = {}
        y_train_encoded = np.zeros_like(Y_train)

        for j in range(Y.shape[1]):
            classes = np.unique(Y_train[:, j])
            mapping = {c: i for i, c in enumerate(classes)}
            self.label_mappings[j] = mapping
            y_train_encoded[:, j] = np.array([mapping[v] for v in Y_train[:, j]])

        return X_train,y_train_encoded,X_test,Y_test

    ## @brief Build the Gaussian Process kernel
    #
    #  Constructs a composite kernel: ConstantKernel * RBF
    def build_model(self):
        self.kernel = ConstantKernel(
            self.constant_value, self.constant_bounds
        ) * RBF(self.rbf_length_scale, self.rbf_bounds)


    ## @brief Train Gaussian Process classifiers on data
    #
    #  Training procedure:
    #  1. Preprocess data 
    #  2. Build kernel
    #  3. Train separate GP classifier for each target variable
    #  4. Store trained models in self.models dictionary
    #
    #  @param X Full input feature array of shape (n_samples, n_features)
    #  @param Y Full target label array of shape (n_samples, n_targets)
    #  @return Tuple (X_test, Y_test) 
    #
    #  @note Each target gets its own GP classifier
    #  @note Training is slow for large datasets, should be used in low data regimes
    def train_model(self, X, Y):

        X_train,Y_train,X_test, Y_test = self.preprocess_data(X, Y)
        self.build_model()

        self.models = {}

        for j in range(Y_train.shape[1]):
            gpc = GaussianProcessClassifier(
                kernel=self.kernel,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.random_state
            )
            gpc.fit(X_train, Y_train[:, j])
            self.models[j] = gpc

        self.is_trained = True
        return X_test, Y_test


    ## @brief Make predictions on new data
    #
    #  Predicts class labels for all targets using trained GP classifiers
    #  Automatically decodes predictions back to original label space
    #
    #  @param X Input features for prediction of shape (n_samples, n_features)
    #  @return Dictionary mapping target names to predicted labels
    #    Format: {f"target_{j}": [predicted_labels]}
    #
    #  @throws ValueError If model has not been trained yet
    def predict(self, X):

        if not self.is_trained:
            raise ValueError("Model not trained")

        results = {}

        for j, model in self.models.items():
            pred_idx = model.predict(X)
            reverse_map = {v: k for k, v in self.label_mappings[j].items()}
            results[f"target_{j}"] = [reverse_map[i] for i in pred_idx]

        return results
    
    ## @brief Predict class probabilities for all targets
    #
    #  Returns probability distributions over classes for each target.
    #  Useful for understanding prediction confidence and uncertainty.
    #
    #  @param X Input features of shape (n_samples, n_features)
    #  @return Dictionary mapping target names to probability arrays
    #    Format: {f"target_{j}": array of shape (n_samples, n_classes_j)}
    #
    #  @throws ValueError If model has not been trained yet
    def predict_proba(self, X):
        """Return prediction probabilities for each target as a dict {target_j: (n_samples, n_classes)}."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        prob_dict = {}

        for j, model in self.models.items():
            prob_dict[f"target_{j}"] = model.predict_proba(X)

        return prob_dict

    ## @brief Save the trained Gaussian Process model to memory
    #
    #  Saves all necessary components for model reconstruction:
    #  - Trained GP classifiers for each target
    #  - Feature scaler
    #  - Label encodings
    #  - Hyperparameter configuration
    #
    #  @param filepath Path where model should be saved (should end with .joblib)
    #
    #  @throws ValueError If model has not been trained yet
    def save(self, filepath: str):
        """
        Save the Gaussian Process model.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")


        model_data = {
            "models": self.models,             
            "scaler": self.scaler,             
            "label_mappings": self.label_mappings,
            "config": {
                "n_restarts_optimizer": self.n_restarts_optimizer,
                "random_state": self.random_state,
                "constant_value": self.constant_value,
                "constant_bounds": self.constant_bounds,
                "rbf_length_scale": self.rbf_length_scale,
                "rbf_bounds": self.rbf_bounds,
            },
            "model_name": self.model_name,
            "is_trained": self.is_trained,
        }
        
        dump(model_data, filepath)
        print(f"GPModel saved to {filepath}")


    ## @brief Load a previously saved Gaussian Process model
    #  Restores a complete model from disk including:
    #  - All trained GP classifiers
    #  - Feature scaler
    #  - Label encodings
    #  - Hyperparameters
    #
    #  @param filepath Path to the saved model file (.joblib)
    def load(self, filepath: str):
        """
        Load a previously saved Gaussian Process model.
        """
    
        model_data = load(filepath)
        

        self.models = model_data["models"]
        self.scaler = model_data["scaler"]
        self.label_mappings = model_data["label_mappings"]
        
        cfg = model_data["config"]
        self.n_restarts_optimizer = cfg["n_restarts_optimizer"]
        self.random_state = cfg["random_state"]
        self.constant_value = cfg["constant_value"]
        self.constant_bounds = cfg["constant_bounds"]
        self.rbf_length_scale = cfg["rbf_length_scale"]
        self.rbf_bounds = cfg["rbf_bounds"]
        
        self.model_name = model_data.get("model_name", "GPModel")
        self.is_trained = model_data.get("is_trained", True)
            
    ## @brief Plot confusion matrices for all targets
    #
    #
    #  @param X Test input features
    #  @param Y True test labels
    #  @param target_names Optional list of descriptive names for each target
    def plot_confusions(self, X, Y, target_names=None):
 
        y_pred_dict = self.predict(X)
        n_targets = len(y_pred_dict)

        fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))
        if n_targets == 1:
            axes = [axes]

        for i, key in enumerate(y_pred_dict.keys()):
            y_true = Y[:, i]
            y_pred = np.array(y_pred_dict[key])

            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", ax=axes[i])

            classes = list(self.label_mappings[i].keys())
            axes[i].set_xticks(range(len(classes)))
            axes[i].set_yticks(range(len(classes)))
            axes[i].set_xticklabels(classes)
            axes[i].set_yticklabels(classes)

            name = target_names[i] if target_names else f"Target {i}"
            axes[i].set_title(f"Confusion Matrix - {name}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")

        plt.tight_layout()
        plt.show()


    ## @brief Compute and display classification metrics
    #
    #  Calculates and prints performance metrics:
    #  - Per-target accuracy
    #  - Classification report (precision, recall, F1-score)
    #  - Overall accuracy (all targets correct simultaneously)
    #
    #  @param X Test input features
    #  @param Y True test labels
    #  @param target_names Optional list of names for each target
    #
    #  @throws ValueError If model has not been trained
    def compute_metrics(self, X, Y, target_names=None):
        """Compute classification metrics per target."""
        if not self.is_trained:
            raise ValueError("Model not trained.")

        y_pred_dict = self.predict(X)
        n_targets = len(y_pred_dict)

        print("\n" + "=" * 50)
        print(f"GAUSSIAN PROCESS METRICS")
        print("=" * 50)

        all_correct = np.ones(len(Y), dtype=bool)

        for i, (key, y_hat) in enumerate(y_pred_dict.items()):
            y_true = Y[:, i]
            y_hat = np.array(y_hat)

            acc = accuracy_score(y_true, y_hat)
            all_correct &= (y_true == y_hat)

            name = target_names[i] if target_names else f"Target {i}"
            print(f"\n{name} Accuracy: {acc:.3f}")
            print(classification_report(y_true, y_hat, zero_division=0))

        print(f"\nOverall Accuracy: {all_correct.mean():.3f}")
        print("=" * 50)


    ## @brief Visualize predictions across feature space
    #
    #  Creates visualization of model predictions on the parameters space
    # 
    #  @param resolution Number of grid points per dimension (default: 100)
    #  @note Only works for 1D or 2D input features
    def visualize_predictions(self, resolution=100):
        if not self.is_trained:
            return

        n_features = len(self.input_min)
        if n_features == 1:
            self._visualize_1d(resolution)
        elif n_features == 2:
            self._visualize_2d(resolution)
        else:
            print("Visualization not supported")

    

    ## @brief Visualize GP predictions for 1D inputs (internal)
    #
    #  Creates two heatmaps per target:
    #  - Top: Predicted class across feature space
    #  - Bottom: Prediction confidence across feature space
    #
    #  @param resolution Grid resolution (resolution x resolution points)
    #
    #  @note This is an internal method, use visualize_predictions() instead
    def _visualize_1d(self, resolution=100):
        """Visualize GP predictions for 1D inputs."""
        x_grid = np.linspace(self.input_min[0], self.input_max[0], resolution)
        X_scaled = self.scaler.transform(x_grid.reshape(-1, 1))
        
        n_targets = len(self.models)
        fig, axes = plt.subplots(2, n_targets, figsize=(5*n_targets, 8))
        if n_targets == 1:
            axes = axes.reshape(2, 1)

        for i, model in self.models.items():
            # Predict classes and probabilities
            classes = model.predict(X_scaled)
            probas = model.predict_proba(X_scaled)
            confidence = probas.max(axis=1)

            # Map back to original labels
            if self.label_mappings and i in self.label_mappings:
                reverse_map = {v: k for k, v in self.label_mappings[i].items()}
                classes = np.array([reverse_map[c] for c in classes])

            # Plot predicted classes
            axes[0, i].plot(x_grid, classes, 'o-', markersize=3)
            axes[0, i].set_title(f'Target {i} Predictions')
            axes[0, i].set_xlabel('Feature')
            axes[0, i].set_ylabel('Class')

            # Plot confidence
            axes[1, i].plot(x_grid, confidence, 'r-', markersize=3)
            axes[1, i].set_title(f'Target {i} Confidence')
            axes[1, i].set_xlabel('Feature')
            axes[1, i].set_ylabel('Confidence')
            axes[1, i].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

    ## @brief Visualize GP predictions for 2D inputs (internal)
    #
    #  Creates two heatmaps per target:
    #  - Top: Predicted class across feature space
    #  - Bottom: Prediction confidence across feature space
    #
    #  @param resolution Grid resolution (resolution x resolution points)
    #
    #  @note This is an internal method, use visualize_predictions() instead
    def _visualize_2d(self, resolution=100):
        """Visualize GP predictions for 2D inputs."""
        x1 = np.linspace(self.input_min[0], self.input_max[0], resolution)
        x2 = np.linspace(self.input_min[1], self.input_max[1], resolution)
        xx, yy = np.meshgrid(x1, x2)
        X_grid = np.column_stack([xx.ravel(), yy.ravel()])
        
        X_scaled = X_grid
        
        n_targets = len(self.models)
        fig, axes = plt.subplots(2, n_targets, figsize=(5*n_targets, 10))
        if n_targets == 1:
            axes = axes.reshape(2, 1)

        for i, model in self.models.items():
            # Predict classes and probabilities
            classes = model.predict(X_scaled)
            probas = model.predict_proba(X_scaled)
            confidence = probas.max(axis=1)

            # Reshape for plotting
            classes = classes.reshape(resolution, resolution)
            confidence = confidence.reshape(resolution, resolution)

            # Map back to original labels
            if self.label_mappings and i in self.label_mappings:
                reverse_map = {v: k for k, v in self.label_mappings[i].items()}
                classes = np.vectorize(reverse_map.get)(classes)

            # Plot predicted classes
            im1 = axes[0, i].imshow(classes, origin='lower',
                                    extent=[self.input_min[0], self.input_max[0],
                                            self.input_min[1], self.input_max[1]],
                                    cmap='coolwarm', aspect='auto')
            axes[0, i].set_title(f'Target {i} Predictions')
            axes[0, i].set_xlabel('Feature 1')
            axes[0, i].set_ylabel('Feature 2')
            plt.colorbar(im1, ax=axes[0, i])

            # Plot confidence
            im2 = axes[1, i].imshow(confidence, origin='lower',
                                    extent=[self.input_min[0], self.input_max[0],
                                            self.input_min[1], self.input_max[1]],
                                    cmap='hot', vmin=0, vmax=1, aspect='auto')
            axes[1, i].set_title(f'Target {i} Confidence')
            axes[1, i].set_xlabel('Feature 1')
            axes[1, i].set_ylabel('Feature 2')
            plt.colorbar(im2, ax=axes[1, i])

        plt.tight_layout()
        plt.show()


