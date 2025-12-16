from modelnn import ModelNN   
from modelGPC import GPModel
from config import load_config
from dataset import MatDataset

if __name__ == "__main__":


    config=load_config("configs/config.json")
    dataset = MatDataset(config["dataset_file"])
    dataset.load()

    X,Y = dataset.prepare_ml_data( config['input_fields'], config['target_fields'])
    target_names = config['target_fields']

    
    model = ModelNN(config['neural_network'])


    X_test,Y_test = model.train_model(X, Y)


    result = model.predict(X_test)
    print(result)

    model.plot_training()
    model.plot_confusions(X_test,Y_test,target_names)
    metrics = model.compute_metrics(X_test,Y_test,target_names)
    model.visualize_predictions()


    model.save("model_family1.pt")

    model2=GPModel(config['gaussian_process'])

    X_test,Y_test=model2.train_model(X, Y)

    result_gp = model2.predict(X_test)
    print(result_gp)
    
    model2.plot_confusions(X_test,Y_test,target_names)
    metrics2=model2.compute_metrics(X_test,Y_test,target_names)
    model2.visualize_predictions()

    model2.save("model_family1_GPC.joblib")

