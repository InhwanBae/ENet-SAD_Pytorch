import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.random.seed(42)


class Metrics:
    '''Compute tpr, fpr, fpr, fnr and balanced accuracy'''

    @classmethod
    def compute_tpr(cls, y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_pos = y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tp / (tp + fn)

    @staticmethod
    def _compute_tpr(y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_pos = y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tp / (tp + fn)

    @classmethod
    def compute_tnr(cls, y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tn / (tn + fp)

    @staticmethod
    def _compute_tnr(y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_pred_neg = 1 - y_pred
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tn / (tn + fp)

    @classmethod
    def compute_ppv(cls, y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_pos = y_pred
        y_true_pos = y_true
        y_true_neg = 1 - y_true

        tp = np.sum(y_pred_pos * y_true_pos + 1e-10)
        fp = np.sum(y_pred_pos * y_true_neg + 1e-10)
        return tp / (tp + fp)

    @classmethod
    def compute_npv(cls, y_true, y_pred):
        y_true = y_true.to('cpu').numpy()
        y_pred = y_pred.to('cpu').numpy()

        y_pred_neg = 1 - y_pred
        y_true_pos = y_true
        y_true_neg = 1 - y_true

        tn = np.sum(y_pred_neg * y_true_neg + 1e-10)
        fn = np.sum(y_pred_neg * y_true_pos + 1e-10)
        return tn / (tn + fn)

    @classmethod
    def balanced_accuracy(cls, y_true, y_pred):
        tpr = cls._compute_tpr(y_true, y_pred)
        tnr = cls._compute_tnr(y_true, y_pred)
        return (tpr + tnr) / 2


def to_one_hot(tensor, nClasses):
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, nClasses, h, w).to(tensor.device).scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=4):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot):
        """
        IoU Loss for individual examples
        inputs - N x {Classes or higher} x H x W
        target_oneHot - N x {Classes or higher} x H x W
        BG can be ignored
        """

        N = inputs.size()[0]
        C = inputs.size()[1]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, C, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, C, -1).sum(2)

        loss = inter / union

        ## Return average loss over classes and batch
        return -(loss[:, -self.classes].mean() - 1.)


'''
def IoULoss(pred, target):
    epsilon = 0.0001
    inter = torch.dot(pred.view(-1), target.view(-1))
    union = torch.sum(pred) + torch.sum(target)
    l_iou = 1 - inter / (union + epsilon)
    l_iou = torch.mean(l_iou)
    return l_iou
'''


class DCLoss(torch.autograd.Function):
    """Dice coeff for individual examples"""

    @staticmethod
    def forward(ctx, pred, target):
        #target = target.type(torch.FloatTensor).to(cfg.device)
        #target = target.type(torch.FloatTensor).cuda()
        target = target.type(torch.FloatTensor).to(pred.device)
        pred = torch.abs(pred)
        eps = 0.0001
        inter = torch.dot(pred.view(-1), target.view(-1))
        union = torch.sum(pred) + torch.sum(target) + eps
        ctx.save_for_backward(pred, target, inter, union)
        t = (2 * inter + eps) / union
        return t

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        pred, target, inter, union = ctx.saved_variables
        grad_input = grad_output * 2 * (target * union - inter) / (union * union)
        return grad_input, None
