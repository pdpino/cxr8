"""
v4 differs from v3 in:
    - uses DenseNet-121 instead of ResNet-50
"""

import torch
import torch.nn as nn
from torchvision import models

class DensenetBasedModel(nn.Module):
    def __init__(self, train_resnet=False, n_diseases=14):
        """.
        
        Note that the parameter is called train_resnet, but in this case Densenet is used (i.e. train_densenet)
        """
        super(DensenetBasedModel, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        
        if not train_resnet:
            for param in self.densenet.parameters():
                param.requires_grad = False

        n_densenet_features = 1024
        n_densenet_output_size = 16 # With input of 512

        self.global_pool = nn.Sequential(
            nn.MaxPool2d(n_densenet_features)
        )
        self.prediction = nn.Sequential(
            nn.Linear(n_densenet_features, n_diseases),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.densenet.features(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        embedding = x
        
        x = self.prediction(x) # n_samples, n_diseases
        
        # TODO
        activations = None
        
        return x, embedding, activations
