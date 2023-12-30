import torch
import torch.nn as nn
from typing import List, Callable, Any


def get_accuracy(model: nn.Module, vloader: torch.utils.data.DataLoader, device: torch.device):
    """
    Calculate the pixel level accuracy over the validation set
    vlodaer: validation set dataloader
    """
    model.eval()
    correct = 0
    total = 0
    for i, (features, label) in enumerate(vloader):
        features = features[0].to(device)
        label = label.to(device)
        out = (model(features) > 0.5).reshape(-1)
        label = (label == 255).reshape(-1)
        correct += (out==label).sum().item()
        total += out.shape[0]
    return correct/total

    model.train()

def add_dimension(tensor : torch.Tensor):
    return tensor.unsqueeze(1)








