"""
v3 differs from v0 in:
    - does not have transition layer
"""

import torch
import torch.nn as nn
from torchvision import models

class ResnetBasedModel(nn.Module):
    def __init__(self, train_resnet=False, n_diseases=14):
        """."""
        super(ResnetBasedModel, self).__init__()
        self.model_ft = models.resnet50(pretrained=True)
        
        if not train_resnet:
            for param in self.model_ft.parameters():
                param.requires_grad = False

        n_resnet_features = 2048
        n_resnet_output_size = 16 # With input of 512

        self.global_pool = nn.Sequential(
            nn.MaxPool2d(n_resnet_output_size)
        )
        self.prediction = nn.Sequential(
            nn.Linear(n_resnet_features, n_diseases),
            nn.Sigmoid()
        )

        self.saved_gradient = None
        self.handler = None
        
    def save_gradient(self, grad):
        self.saved_gradient = grad
        self.handler.remove()

        return None
        
    def forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x) # n_samples, 2048, height=16, width=16

        # print("After resnet: ", x.size())
        
        # FIXME
#         x.retain_grad()
#         print(x.retains_grad)
#         handler = x.register_hook(self.save_gradient)
        
        pred_weights, pred_bias_unused = list(self.prediction.parameters()) # size: n_diseases, n_features = 2048
        # x: activations from prev layer # size: n_samples, n_features, height = 16, width = 16
        # bbox: for each sample, multiply n_features dimensions
        # --> activations: n_samples, n_diseases, height, width
        activations = torch.matmul(pred_weights, x.transpose(1, 2)).transpose(1, 2)
        
        x = self.global_pool(x)
        
        # print("After global pool: ", x.size())

        x = x.view(x.size(0), -1)
        
        # print("After view: ", x.size())
        
        embedding = x
        
        x = self.prediction(x) # n_samples, n_diseases
        
        return x, embedding, activations

