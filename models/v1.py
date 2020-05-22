"""
v1 differs from v0 in:
    - has two FC layers at the end
    - (for now) does not calculate CAM (since it cannot be applied directly) (TODO: apply GradCAM)
"""

import torch
import torch.nn as nn
from torchvision import models

class ResnetBasedModel(nn.Module):
    def __init__(self, train_resnet=False, n_diseases=14, n_features=2048, n_hidden=100):
        """.
        
        params:
          n_features: Number of features from ImageNet model (Resnet-50 in this case)
        """
        super(ResnetBasedModel, self).__init__()
        self.model_ft = models.resnet50(pretrained=True)
        
        if not train_resnet:
            for param in self.model_ft.parameters():
                param.requires_grad = False

        # REVIEW: is this layer ok?
        self.transition = nn.Sequential(
            nn.Conv2d(n_features, 2048, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.global_pool = nn.Sequential(
            nn.MaxPool2d(16) # 32
        )
        self.prediction = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Linear(n_hidden, n_diseases),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x) # n_samples, n_features = 2048, height, width

        # print("Before transition: ", x.size())

        x = self.transition(x)
        
        # print("After transition: ", x.size())

        activations = None

        x = self.global_pool(x)
        
        # print("After global pool: ", x.size())

        x = x.view(x.size(0), -1)
        
        # print("After view: ", x.size())
        
        embedding = x # TODO: embedding should be between the two layers?
        
        x = self.prediction(x) # n_samples, n_diseases
        
        return x, embedding, activations

