import torch
from torch import nn


class RegressionLoss(nn.Module):

    def __init__(self):
        super(RegressionLoss, self).__init__()

    def forward(self, outputs, targets):
        num_classes = outputs.size(1)
        if num_classes > 1:
            outputs = (outputs.softmax(dim=1) * torch.arange(1, num_classes + 1)).sum(dim=1)
        return nn.functional.mse_loss(outputs, targets)

    @staticmethod
    def accuracy(outputs: torch.FloatTensor, targets: torch.FloatTensor) -> \
            (torch.FloatTensor, torch.IntTensor, torch.FloatTensor):
        with torch.no_grad():
            probs = outputs
            pred = outputs
            num_classes = outputs.size(1)
            if num_classes > 1:
                probs = outputs.softmax(dim=1)
                _, pred = outputs.max(dim=1)
                outputs = probs * torch.arange(1, num_classes + 1)
                acc = (outputs.sum(dim=1) - targets).abs().mean()
            else:
                acc = (outputs - targets).abs().mean()
            return acc, pred, probs


if __name__ == '__main__':
    loss_module = RegressionLoss()
    # my_outputs = torch.rand(10, 5)
    my_outputs = torch.rand(10, 1)
    my_targets = torch.rand(10,1)
    my_loss = loss_module(my_outputs, my_targets)
    print(f'my_loss: {my_loss}')
    my_acc, my_pred, my_probs = loss_module.accuracy(my_outputs, my_targets)
    print(f'acc: {my_acc.detach().cpu().numpy()}, \n'
          f'pred: {my_pred.detach().cpu().numpy()}, \n'
          f'my_probs: {my_probs.detach().cpu().numpy()}')
