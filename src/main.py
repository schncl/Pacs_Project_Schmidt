from modelnn import ModelNN   
from modelGPC import GPModel
from config import load_config
from dataset import MatDataset
import argparse



def parse_args():
    """Parser for command line arguments"""
    parser=argparse.ArgumentParser(description="ML training specs")
    parser.add_argument(
        '--config',
        type=str,
        default="../configs/config_heat.json"
    )
    return parser.parse_args()


def main():

    args=parse_args()
    config=load_config(args.config)

    dataset = MatDataset(config["dataset_path"])
    dataset.load()

    X,Y = dataset.prepare_ml_data( config['input_fields'], config['target_fields'])
    target_names = config['target_fields']

    if config["model"] in ["nn","both"]:
    
        model = ModelNN(config['neural_network'])


        X_test,Y_test = model.train_model(X, Y)

    
        if  config["plots"]:
            model.plot_training()
            model.plot_confusions(X_test,Y_test,target_names)
            model.visualize_predictions()
        
        metrics = model.compute_metrics(X_test,Y_test,target_names)
        if config["save_model"]:
            model.save(config["path_to_model_nn"])
    
    if config["model"] in ["gp","both"]:


        model2=GPModel(config['gaussian_process'])
        if X.shape[0] >= config["samples_gp"]:
            X = X[:config["samples_gp"], :]
            Y = Y[:config["samples_gp"], :]
        else:
            raise ValueError(f"Not enough samples. Need {config['samples_gp']}, but only have {X.shape[0]}")

        X_test,Y_test=model2.train_model(X, Y)
        
        if  config["plots"]:
            model2.plot_confusions(X_test,Y_test,target_names)
            metrics2=model2.compute_metrics(X_test,Y_test,target_names)
            model2.visualize_predictions()
                

        if config["save_model"]:
            model2.save(config["path_to_model_gp"])


if __name__ == "__main__":
    main()

