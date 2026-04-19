import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file)
    return config