from .resnet import load_resnet
from .mobilenet import load_mobilenet

def load_model(name):
    if 'resnet' in name:
        return load_resnet(name)
    elif 'mobilenet' in name:
        return load_mobilenet()
    else:
        raise ValueError("invalid network name")