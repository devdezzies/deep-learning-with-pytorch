"""
contains pytorch model code to instantiate a TinyVGG model.
"""
import torch 
from torch import nn 
import torchvision

class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                     out_channels=hidden_units,
                     kernel_size=3, 
                     stride=1, 
                     padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                     out_channels=hidden_units,
                     kernel_size=3, 
                     stride=1, 
                     padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, 
                        stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0), 
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, out_features=output_shape)
        )
    def forward(self, x):
        z = self.conv_block_1(x)
        z = self.conv_block_2(z)
        z = self.classifier(z)
        return z

def create_model_baseline_effnetb0(out_feats: int, device: torch.device = None) -> torch.nn.Module:
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # change the output layer 
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=out_feats, 
                        bias=True)).to(device)
    
    return model