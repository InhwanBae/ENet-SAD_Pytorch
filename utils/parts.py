import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import function
from torchvision import models
import numpy as np
#from config import Config

np.random.seed(42)

#cfg = Config()


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


# class DCLoss(torch.autograd.Function):

#     def __init__(self):
#         super().__init__()

#     @staticmethod
#     def forward(ctx, pred, label):
#         label = torch.cat((1. - torch.unsqueeze(label, 1), torch.unsqueeze(label, 1)), 1).type(torch.FloatTensor).to(cfg.device)
#         loss = 2. * torch.sum(torch.abs(pred * label)) / torch.sum(torch.abs(pred) + torch.abs(label))
#         ctx.save_for_backward(pred, label)
#         return loss

#     @staticmethod
#     def backward(ctx, grad_output):
#         pred, label = ctx.saved_tensors
#         dDice = torch.add(torch.mul(label, 2), torch.mul(pred, -4))
#         # grad_input = torch.cat((torch.mul(torch.unsqueeze(dDice,1), grad_output.item()),\
#         #     torch.mul(torch.unsqueeze(dDice,1), -grad_output.item())), dim = 1)
#         grad_input = torch.mul(dDice, -grad_output.item())
#         return grad_input, None

def bivariate_loss(V_pred, V_trgt):
    # mux, muy, sx, sy, corr
    # assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr

    sxsy = sx * sy

    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    # result = -torch.log(torch.clamp(result, min=epsilon, max=1))
    result = torch.mean(result)

    return result


def to_one_hot(tensor, nClasses):
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, nClasses, h, w).cuda().scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=4):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot):
        """
        IoU Loss for individual examples
        inputs - N x Classes x H x W
        target_oneHot - N x Classes x H x W
        """

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union

        ## Return average loss over classes and batch
        return 1 - loss.mean()

'''
def IoULoss(pred, target):
    """IoU Loss for individual examples"""


    epsilon = 0.0001
    inter = torch.dot(pred.view(-1), target.view(-1))
    union = torch.sum(pred) + torch.sum(target)
    t = 1 - inter / (union + epsilon)
    l_iou = -torch.log(torch.clamp(t, min=epsilon))
    l_iou = torch.mean(l_iou)
    return l_iou
'''

class DCLoss(torch.autograd.Function):
    """Dice coeff for individual examples"""

    @staticmethod
    def forward(ctx, pred, target):
        #target = target.type(torch.FloatTensor).to(cfg.device)
        target = target.type(torch.FloatTensor).cuda()
        pred = torch.abs(pred)
        eps = 0.0001
        # print('input into dice', input.view(-1).size())
        # print('target into dice', target.view(-1).size())
        # inter = torch.sum(torch.mul(pred, target))
        inter = torch.dot(pred.view(-1), target.view(-1))
        union = torch.sum(pred) + torch.sum(target) + eps
        ctx.save_for_backward(pred, target, inter, union)
        t = (2 * inter + eps) / union
        return t

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        pred, target, inter, union = ctx.saved_variables
        grad_input = grad_output * 2 * (target * union - inter) \
                     / (union * union)
        return grad_input, None


# class DCLoss(nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, pred, label):
#         label = torch.cat((1. - torch.unsqueeze(label, 1), torch.unsqueeze(label, 1)), 1).type(torch.FloatTensor).to(cfg.device)
#         loss = 2. * torch.sum(torch.abs(pred * label)) / torch.sum(torch.abs(pred) + torch.abs(label))
#         return loss

if __name__ == '__main__':
    test_value = torch.ones((2, 2, 64, 64), dtype=torch.float, requires_grad=True)
    test_value_ = torch.ones((2, 64, 64), dtype=torch.float)
    criterion = DCLoss()
    loss = criterion(test_value, test_value_)
    torch.autograd.gradcheck(criterion, (test_value, test_value_))
    # loss.backward()
    # layer = slice()
    # ans = layer(test_value)
    print()
