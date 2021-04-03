import torch
from torch import nn


class RankingLoss(nn.Module):
    def __init__(self,args):
        super(RankingLoss, self).__init__()
        self.margin = 0.0
        self.target = torch.ones(args.batch_size)
        if args.gpu:
            self.target = self.target.cuda(0)

    def forward(self, image_p_batch, _):
        image_p_batch = image_p_batch.view(-1).sigmoid()
        loss = nn.functional.margin_ranking_loss(image_p_batch[0::2], image_p_batch[1::2],
                                                 self.target, margin=self.margin)
        return loss

    @staticmethod
    def accuracy(outputs: torch.FloatTensor, _) -> \
            (torch.float32, torch.BoolTensor, torch.FloatTensor):
        with torch.no_grad():
            outputs = outputs.view(-1).sigmoid()
            probs = torch.stack([outputs[0::2], outputs[1::2]], dim=1)
            pred = (outputs[0::2] - outputs[1::2]) > 0
            acc = pred.int().float().mean() * 100.0
            return acc, pred, probs

