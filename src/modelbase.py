from abc import ABC, abstractmethod



class BaseModel(ABC):
    """Abstract Class for models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.is_trained = False
        self.history = {}
        
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train_model(self, X_train, y_train):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X) :
        """Make predictions"""
        pass
    
    @abstractmethod
    def preprocess_data(self, X, Y) :
        """Preprocessing pipeline"""
        pass

    @abstractmethod
    def save(self,filepath):
        """Model Persistence"""
        pass
    
    @abstractmethod
    def load(self,filepath):
        """Loading existing model"""
        pass


