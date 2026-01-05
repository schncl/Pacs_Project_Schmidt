## @file modelbase.py
#  @brief Abstract base class for machine learning models
#
#  This module defines the BaseModel abstract class which serves as
#  the interface that all model implementations must follow



from abc import ABC, abstractmethod


## @class BaseModel
#  @brief Abstract base class for ML models
#
#  This class defines the common interface for all machine learning
#  models in the framework. All concrete model implementations must
#  inherit from this class and implement its abstract methods
class BaseModel(ABC):
    """Abstract Class for models"""
    
    ## @brief Constructor for BaseModel
    #
    #  Initializes the common attributes that all models share
    #
    #  @param model_name String identifier for the model 

    def __init__(self, model_name):
       
       
        ## @var model_name
        #  @brief String identifier for this model instance
        self.model_name = model_name
        

        ## @var is_trained
        #  @brief Boolean flag indicating whether the model has been trained
        #  Set to False on initialization and True after training
        self.is_trained = False


        ## @var history
        #  @brief Dictionary storing training history and metrics
        self.history = {}
    


    ## @brief Build the model architecture
    #
    #  Abstract method that must be implemented by subclasses to
    #  construct the model architecture
    #
    #  @note Must be implemented by subclasses
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass



    ## @brief Train the model on data
    #
    #  Abstract method that must be implemented by subclasses to
    #  train the model using the provided training data
    #
    #  @param X_train Training input features of shape (n_samples, n_features)
    #  @param y_train Training target labels of shape (n_samples, n_targets)
    #  @return Returns (X_test, y_test) 
    #  @note Must be implemented by subclasses
    @abstractmethod
    def train_model(self, X_train, y_train):
        """Train the model"""
        pass



    ## @brief Make predictions on new data
    #
    #  Abstract method that must be implemented by subclasses to
    #  generate predictions for new samples
    #
    #  @param X Input features for prediction of shape (n_samples, n_features)
    #  @return Model predictions
    #  @note Must be implemented by subclasses
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass



    ## @brief Preprocess input data
    #
    #  Abstract method that must be implemented by subclasses to
    #  perform data preprocessing steps before training
    #
    #  @param X Input features array of shape (n_samples, n_features)
    #  @param Y Target labels array of shape (n_samples, n_targets)
    #  @return Preprocessed data
    #  @note Must be implemented by subclasses
    @abstractmethod
    def preprocess_data(self, X, Y):
        """Preprocessing pipeline"""
        pass



    ## @brief Save model to memory
    #
    #  Abstract method that must be implemented by subclasses to
    #  save the trained model to a file for later use
    #
    #  @param filepath Path where model should be saved
    #  @note Must be implemented by subclasses
    @abstractmethod
    def save(self, filepath):
        """Model Persistence"""
        pass



    ## @brief Load model from disk
    #
    #  Abstract method that must be implemented by subclasses to
    #  load a previously saved model from a file
    #
    #  @param filepath Path to saved model file
    #
    #  @note Must be implemented by subclasses
    @abstractmethod
    def load(self, filepath):
        """Loading existing model"""
        pass