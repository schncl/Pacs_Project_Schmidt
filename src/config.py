## @file config.py
#  @brief Configuration file loader for JSON configurations
#
#  This module provides functionality to load and parse JSON
#  configuration files for the ML training pipeline

import json


## @brief Load configuration from JSON file
#
#  Reads a JSON configuration file and returns it as a Python dictionary
#
#  @param config_path Path to the JSON configuration file (default: "config.json")
#  @return Dictionary containing the configuration parameters
#
def load_config(config_path="../configs/config.json"):
    """Simple function to load JSON config""" 
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
