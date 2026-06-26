import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from config import NUM_CLASSES

def create_model():
    model=resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features=model.fc.in_features
    model.fc=nn.Linear(num_features,NUM_CLASSES)
    return model

if __name__ == "__main__":
    model=create_model()
    print(model.fc)