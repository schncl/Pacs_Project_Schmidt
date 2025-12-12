from modelbase import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler




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


