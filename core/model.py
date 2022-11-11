import torch
import torch.nn as nn
import torch.nn.functional as F 

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Identity()
        )
        
    def forward(self, x):
        return self.pipeline(x)
