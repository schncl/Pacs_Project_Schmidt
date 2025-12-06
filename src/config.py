import json

def load_config(config_path="config.json"):
    """Simple function to load JSON config""" 
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
