import torch
import torch.nn as nn
from torchvision import models
import numpy as np # DEBUG

class ResnetBasedModel(nn.Module):
    def __init__(self, train_resnet=False, n_diseases=14, n_features=2048):
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
            nn.MaxPool2d(32) # 32 is the maximum value of S --> Global Pooling
        )
        self.prediction = nn.Sequential(
            nn.Linear(n_features, n_diseases),
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

#         print("Before transition: ", x.size())

#         y = x # DEBUG
        
        x = self.transition(x)
        
        # DEBUG
#         eps = 1e-05
#         x2 = x / torch.abs(x + eps)
#         y2 = y / torch.abs(y + eps)
#         sign_shift = ((x2 * y2 + 1) <= eps).sum().item()
#         tot = np.prod(y.size())
#         print("Sign shift: ", sign_shift, sign_shift / tot)
#         print("Diff: ", (x - y).mean().item())
#         y[y==0] = x[y==0]
#         print("Fraction: ", (torch.abs(x)/torch.abs(y)).mean().item())
        # END DEBUG
        
#         print("After transition: ", x.size())
        
        pred_weights, pred_bias_unused = list(self.prediction.parameters()) # size: n_diseases, n_features = 2048
        # x: activations from prev layer # size: n_samples, n_features, height = 16, width = 16
        # bbox: for each sample, multiply n_features dimensions
        # --> activations: n_samples, n_diseases, height, width
        activations = torch.matmul(pred_weights, x.transpose(1, 2)).transpose(1, 2)
        
        # print("\tweights: ", pred_weights.size())
        # print("\tTransposed 0,1: ", x.transpose(0, 1).size())
        # print("\tTransposed 1,2: ", x.transpose(1, 2).size())
        
        x = self.global_pool(x)
        
#         print("After global pool: ", x.size())

        x = x.view(x.size(0), -1)
        
#         print("After view: ", x.size())
        
        embedding = x
        
        x = self.prediction(x) # n_samples, n_diseases
        
        return x, embedding, activations

