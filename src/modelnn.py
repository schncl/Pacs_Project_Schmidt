from modelbase import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns




class MultiTargetClassifier(nn.Module):
    """Simple NN for multiclass classification"""
    def __init__(self, input_dim, n_classes_list, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)
        self.heads = nn.ModuleList(nn.Linear(prev_dim, i) for i in n_classes_list)
    
    def forward(self, x):
        x = self.shared(x)
        outputs=[]
        for head in self.heads:
            outputs.append(head(x))
        return tuple(outputs)


class ModelNN(BaseModel):
    """This class is the actual classifier"""
    def __init__(self,config):
        super().__init__(model_name="ModelNN")

        self.epochs = config.get("epochs", 300)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.hidden_layers = config.get("hidden_layers", [64, 32])
        self.dropout_rate = config.get("dropout_rate", 0.3)
        self.optimizer_name = config.get("optimizer", "adam")
        self.loss_function = config.get("loss_function", "cross_entropy")


        self.model = None
        self.scaler = None
        self.loader=None
        self.label_mappings = None
        self.input_min = None
        self.input_max = None


    def preprocess_data(self, X, Y):
        """Apply preprocessing to Data and returns torch Dataloader object"""
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]

        #For visualization later on
        self.input_min = X.min(axis=0)
        self.input_max = X.max(axis=0)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_tensor = torch.FloatTensor(X_train_scaled)

        # Labels Encoding
        self.label_mappings = {}
        y_train_list = []

        for j in range(Y.shape[1]):
            classes = np.unique(Y_train[:, j])
            mapping = {c: idx for idx, c in enumerate(classes)}
            self.label_mappings[j] = mapping
            y_train_list.append(torch.LongTensor([mapping[v] for v in Y_train[:, j]]))

        y_train_tensor = torch.stack(y_train_list, dim=1)

        # Torch Infrastructure
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.loader=train_loader

    
        return  X_train,y_train_tensor,X_test,Y_test

    

    def build_model(self, input_dim, num_classes_list):
        """Build the NN """
        self.model = MultiTargetClassifier(
            input_dim=input_dim,
            n_classes_list=num_classes_list,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate
        )



    def train_model(self, X, Y):
        """Train the neural network"""

        # Check wether cuda is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Preprocess data
        _,_,X_test, Y_test = self.preprocess_data(X, Y)

        # Build model
        input_dim = X.shape[1]
        num_classes_list = [len(self.label_mappings[i]) for i in sorted(self.label_mappings.keys())]
        self.build_model(input_dim, num_classes_list)
        self.model.to(device)


        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_func = nn.CrossEntropyLoss()

        self.model.train()
        train_losses = []


        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_Y in self.loader:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)

                total_loss = sum(loss_func(out, batch_Y[:, i]) for i, out in enumerate(outputs))
                avg_loss = total_loss / len(outputs)
                avg_loss.backward()
                optimizer.step()

                epoch_loss += avg_loss.item()

            avg_epoch_loss = epoch_loss / len(self.loader)
            train_losses.append(avg_epoch_loss)

  
            if (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {avg_epoch_loss:.4f}')

        self.history['train_losses'] = train_losses
        self.is_trained = True

        return X_test, Y_test




    def predict(self, X):
        """Make predictions on new data"""
       
        if not self.is_trained:
            raise ValueError("Model not trained")

        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            outputs = self.model(X_tensor)

        # Convert outputs to original labels
        result = {}
        for i, pred in enumerate(outputs):
            predicted_classes = torch.argmax(pred, dim=1).numpy()
            reverse_mapping = {v: k for k, v in self.label_mappings[i].items()}
            result[f'target_{i}'] = [reverse_mapping[c] for c in predicted_classes]

        return result


    def save(self, filepath):
        """Save the neural network weights"""
        if self.model is None:
            raise ValueError("No model to save.")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'label_mappings': self.label_mappings,
            'config': {
                'hidden_layers': self.hidden_layers,
                'dropout_rate': self.dropout_rate,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'optimizer_name': self.optimizer_name,
                'loss_function': self.loss_function
            }
        }

        torch.save(checkpoint, filepath)



    def load(self, filepath):
        """Load a saved model"""

        checkpoint = torch.load(filepath, map_location='cpu')  


        config = checkpoint['config']
        label_mappings = checkpoint['label_mappings']
        num_classes_list = [len(label_mappings[i]) for i in sorted(label_mappings.keys())]

     
        shared_weights_key = next((k for k in checkpoint['model_state_dict'] if 'shared' in k and 'weight' in k), None)
        input_dim = checkpoint['model_state_dict'][shared_weights_key].shape[1]


        self.model = MultiTargetClassifier(
            input_dim=input_dim,
            n_classes_list=num_classes_list,
            hidden_layers=config['hidden_layers'],
            dropout_rate=config['dropout_rate']
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.scaler = checkpoint['scaler']
        self.label_mappings = label_mappings
        self.hidden_layers = config['hidden_layers']
        self.dropout_rate = config['dropout_rate']
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 300)
        self.optimizer_name = config.get('optimizer_name', 'adam')
        self.loss_function = config.get('loss_function', 'cross_entropy')
        self.is_trained = True

    def plot_training(self):
        """Plot of the training loss"""

        plt.plot(self.history['train_losses'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()


    def plot_confusions(self, X_test, Y_test, target_names=None):
        """Plot confusion matrices"""
        
        if not self.is_trained:
            raise ValueError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X_test)
            X_tensor = torch.FloatTensor(X_scaled)
            predictions = self.model(X_tensor)

        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        n_targets = len(predictions)
        fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4))
        if n_targets == 1:
            axes = [axes]

        for i, (pred, ax) in enumerate(zip(predictions, axes)):
            y_true = Y_test[:, i].astype(int)
            y_pred = torch.argmax(pred, dim=1).numpy().astype(int)


            if self.label_mappings and i in self.label_mappings:
                reverse_mapping = {v: k for k, v in self.label_mappings[i].items()}
                y_true = np.array([reverse_mapping[v] if v in reverse_mapping else v for v in y_true])
                y_pred = np.array([reverse_mapping[v] if v in reverse_mapping else v for v in y_pred])

            classes = np.unique(np.concatenate([y_true, y_pred]))
            cm = confusion_matrix(y_true, y_pred, labels=classes)

            sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                        xticklabels=classes, yticklabels=classes,
                        cmap="Blues")

            name = target_names[i] if target_names else f"Target {i}"
            ax.set_title(f'Confusion Matrix - {name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.tight_layout()
        plt.show()



    def compute_metrics(self, X_test, Y_test, target_names=None):
        """Compute accuracy and classification reports"""
      
        if not self.is_trained:
            raise ValueError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X_test)
            X_tensor = torch.FloatTensor(X_scaled)
            predictions = self.model(X_tensor)

        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        accuracies = []
        all_correct = np.ones(len(X_test), dtype=bool)

        print("\n" + "=" * 50)
        print("NEURAL NETWORK METRICS")
        print("=" * 50)

        for i, pred in enumerate(predictions):
            y_true = Y_test[:, i].astype(int)
            y_pred = torch.argmax(pred, dim=1).numpy().astype(int)

         
            if self.label_mappings and i in self.label_mappings:
                reverse_mapping = {v: k for k, v in self.label_mappings[i].items()}
                y_true = np.array([reverse_mapping.get(v, v) for v in y_true])
                y_pred = np.array([reverse_mapping.get(v, v) for v in y_pred])

            accuracy = accuracy_score(y_true, y_pred)
            accuracies.append(accuracy)
            all_correct &= (y_true == y_pred)

            name = target_names[i] if target_names else f"Target_{i}"
            print(f"\n{name} Accuracy: {accuracy:.3f}")
            print(f"{name} Classification Report:")
            print(classification_report(y_true, y_pred, zero_division=0))

        overall_accuracy = all_correct.mean()
        print(f"\nOverall Accuracy (all targets correct per sample): {overall_accuracy:.3f}")
        print("=" * 50)

        return {'accuracies': accuracies, 'overall_accuracy': overall_accuracy}


    def visualize_predictions(self, resolution=100):
        """Visualize NN predictions for 1D or 2D inputs."""
        if not self.is_trained:
            print("Model must be trained before visualization.")
            return

        n_features = len(self.input_min)
        if n_features == 1:
            self._visualize_1d(resolution)
        elif n_features == 2:
            self._visualize_2d(resolution)
        else:
            print(f"Visualization not supported for {n_features} features.")


    def _visualize_1d(self, resolution=100):
        x_grid = np.linspace(self.input_min[0], self.input_max[0], resolution)
        X_scaled = self.scaler.transform(x_grid.reshape(-1, 1))
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        
        n_targets = len(outputs)
        fig, axes = plt.subplots(2, n_targets, figsize=(5*n_targets, 8))
        if n_targets == 1:
            axes = axes.reshape(2, 1)

        for i, pred in enumerate(outputs):
            probs = torch.softmax(pred, dim=1).numpy()
            classes = probs.argmax(axis=1)
            confidence = probs.max(axis=1)

            
            if self.label_mappings and i in self.label_mappings:
                reverse_map = {v: k for k, v in self.label_mappings[i].items()}
                classes = np.array([reverse_map[c] for c in classes])

        
            axes[0, i].plot(x_grid, classes, 'o-', markersize=3)
            axes[0, i].set_title(f'Target {i} Predictions')
            axes[0, i].set_xlabel('Feature')
            axes[0, i].set_ylabel('Class')

        
            axes[1, i].plot(x_grid, confidence, 'r-', markersize=3)
            axes[1, i].set_title(f'Target {i} Confidence')
            axes[1, i].set_xlabel('Feature')
            axes[1, i].set_ylabel('Confidence')
            axes[1, i].set_ylim(0, 1)

        plt.tight_layout()
        plt.show()


    def _visualize_2d(self, resolution=100):
        x1 = np.linspace(self.input_min[0], self.input_max[0], resolution)
        x2 = np.linspace(self.input_min[1], self.input_max[1], resolution)
        xx, yy = np.meshgrid(x1, x2)
        X_grid = np.column_stack([xx.ravel(), yy.ravel()])
        
        X_scaled = self.scaler.transform(X_grid)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

        n_targets = len(outputs)
        fig, axes = plt.subplots(2, n_targets, figsize=(5*n_targets, 10))
        if n_targets == 1:
            axes = axes.reshape(2, 1)

        for i, pred in enumerate(outputs):
            probs = torch.softmax(pred, dim=1).numpy()
            classes = probs.argmax(axis=1).reshape(resolution, resolution)
            confidence = probs.max(axis=1).reshape(resolution, resolution)


            if self.label_mappings and i in self.label_mappings:
                reverse_map = {v: k for k, v in self.label_mappings[i].items()}
                classes = np.vectorize(reverse_map.get)(classes)

          
            im1 = axes[0, i].imshow(classes, origin='lower',
                                    extent=[self.input_min[0], self.input_max[0],
                                            self.input_min[1], self.input_max[1]],
                                    cmap='coolwarm', aspect='auto')
            axes[0, i].set_title(f'Target {i} Predictions')
            axes[0, i].set_xlabel('Feature 1')
            axes[0, i].set_ylabel('Feature 2')
            plt.colorbar(im1, ax=axes[0, i])

         
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

