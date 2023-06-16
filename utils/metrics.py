import torch

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total_samples = labels.size(0)
    correct_predictions = (predicted == labels).sum().item()
    return correct_predictions / total_samples * 100
