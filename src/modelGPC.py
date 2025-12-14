from modelbase import BaseModel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from joblib import dump,load
import numpy as np


class GPModel(BaseModel):
    """Gaussian Process multi-target classifier"""

    def __init__(self, config, model_name="GPModel"):
        
        super().__init__(model_name)

        self.n_restarts_optimizer = config.get("n_restarts_optimizer", 10)
        self.random_state = config.get("random_state", 42)

        self.constant_value = config.get("constant_value", 1.0)
        self.constant_bounds = tuple(config.get("constant_bounds", [1e-3, 1e3]))
        self.rbf_length_scale = config.get("rbf_length_scale", 1.0)
        self.rbf_bounds = tuple(config.get("rbf_bounds", [1e-2, 1e2]))

        self.models = {}
        self.scaler = None
        self.label_mappings = {}
        self.input_min = None
        self.input_max = None


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


    def build_model(self):

        self.kernel = ConstantKernel(
            self.constant_value, self.constant_bounds
        ) * RBF(self.rbf_length_scale, self.rbf_bounds)



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



    def predict(self, X):

        if not self.is_trained:
            raise ValueError("Model not trained")

        #X_scaled = self.scaler.transform(X)
        X_scaled=X
        results = {}

        for j, model in self.models.items():
            pred_idx = model.predict(X_scaled)
            reverse_map = {v: k for k, v in self.label_mappings[j].items()}
            results[f"target_{j}"] = [reverse_map[i] for i in pred_idx]

        return results
    

    def predict_proba(self, X):
        """Return prediction probabilities for each target as a dict {target_j: (n_samples, n_classes)}."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        #X_scaled = self.scaler.transform(X)
        X_scaled=X
        prob_dict = {}

        for j, model in self.models.items():
            prob_dict[f"target_{j}"] = model.predict_proba(X_scaled)

        return prob_dict


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
            

        
