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

#         # To save gradients
#         self.saved_gradient = None
#         self.handler = None
        
#     def save_gradient(self, grad):
#         self.saved_gradient = grad
#         self.handler.remove()

#         return None
        
    def forward(self, x):
        # FIXME: this model won't train if returns one value! needs to return 3 values 
        
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x) # n_samples, 2048, height=16, width=16

        # print("After resnet: ", x.size())
        
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
        
#         return x
        return x, embedding, activations

    def forward_with_cam(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x) # n_samples, 2048, height=16, width=16

        
        pred_weights, pred_bias_unused = list(self.prediction.parameters()) # size: n_diseases, n_features = 2048
        # x: activations from prev layer # size: n_samples, n_features, height = 16, width = 16
        # bbox: for each sample, multiply n_features dimensions
        # --> activations: n_samples, n_diseases, height, width
        activations = torch.matmul(pred_weights, x.transpose(1, 2)).transpose(1, 2)
        
        x = self.global_pool(x)
        
        x = x.view(x.size(0), -1)
        
        embedding = x
        
        x = self.prediction(x) # n_samples, n_diseases
        
        return x, embedding, activations
        

    
    def forward_with_gradcam(self, x, disease_index=0):
        # Set in evaluation mode
        self.eval()

        # Forward pass
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x) # n_samples, 2048, height=16, width=16

        heatmaps = x
        print("heatmap: ", heatmaps.size())
        
        saved_gradients = [] # Use a list to avoid global
        def save_gradient(grad):
            saved_gradients.append(grad)
            return None

        x.requires_grad = True
        handler = x.register_hook(save_gradient)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)        
        output = self.prediction(x) # n_samples, n_diseases
        
        one_hot = torch.zeros(output.size()).to(output.device)
        one_hot[:, disease_index] = 1
        
        output.backward(one_hot)
        
        print("AMOUNT OF GRADIENTS: ", len(saved_gradients))
        
        gradient = saved_gradients[0]
        avg_gradient = gradient.mean(-1).mean(-1) # Average thru space (height and width)

        
        n_samples, n_features, height, width = heatmaps.size()
        result = torch.zeros(n_samples, height, width).to(heatmaps.device)

        for i_sample in range(n_samples):
            alpha = avg_gradient[i_sample]
            heatmap = heatmaps[i_sample]

            for i_feature in range(n_features):
                result[i_sample] += alpha[i_feature] * heatmap[i_feature]
        
        # TODO: pass result through relu
        
        # Clear handler
        handler.remove()
        
        return result

#         for i_disease in range(n_diseases):
#             # Create one-hot vector
#             one_hot = np.zeros(n_samples, n_diseases, requires_grad=True).to(output.device)
#             one_hot[:, i_disease] = 1

#             # Zero grad every part of the model
#             self.model_ft.zero_grad()
#             self.global_pool.zero_grad()
#             self.prediction.zero_grad()

#             # Run backward pass
#             added_one_hot = torch.sum(one_hot * output)
#             added_one_hot.backward()
            
#             # Retain variables!
#             # TODO
