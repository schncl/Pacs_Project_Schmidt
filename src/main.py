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
        default="../configs/config.json"
    )
    parser.add_argument(
        "--plots",
        default=False
    )
    return parser.parse_args()


def main():

    args=parse_args()
    config=load_config(args.config)

    dataset = MatDataset(config["dataset_path"])
    dataset.load()

    X,Y = dataset.prepare_ml_data( config['input_fields'], config['target_fields'])
    target_names = config['target_fields']

    
    model = ModelNN(config['neural_network'])


    X_test,Y_test = model.train_model(X, Y)

  
    if  args.plots:
        model.plot_training()
        model.plot_confusions(X_test,Y_test,target_names)
        model.visualize_predictions()
    
    metrics = model.compute_metrics(X_test,Y_test,target_names)

    model.save("model_family1.pt")
    
    model2=GPModel(config['gaussian_process'])

    X_test,Y_test=model2.train_model(X, Y)

    if not args.noplots:
        model2.plot_confusions(X_test,Y_test,target_names)
        metrics2=model2.compute_metrics(X_test,Y_test,target_names)
        model2.visualize_predictions()
            
    metrics2=model2.compute_metrics(X_test,Y_test,target_names)
    model2.save("model_family1_GPC.joblib")


if __name__ == "__main__":
    main()

