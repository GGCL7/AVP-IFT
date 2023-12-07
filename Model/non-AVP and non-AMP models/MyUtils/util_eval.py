from test_other.configuration import config as cf
import torch
def evaluate_accuracy(data_iter, net):
    device = torch.device("cpu")
    acc_sum, n = 0.0, 0
    for x, y, z in data_iter:
        x, y = x.to(device), y.to(device)
        outputs = net.trainModel(x)

        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n