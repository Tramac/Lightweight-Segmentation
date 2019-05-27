"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss']


# TODO: optim function
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=False, aux_weight=0.4, weight=None, min_kept=100000, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(min_kept=min_kept, ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))
