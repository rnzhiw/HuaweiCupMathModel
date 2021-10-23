import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, max_len=-1):
        self.val = []
        self.count = []
        self.max_len = max_len
        self.avg = 0

    def update(self, val, n=1):
        self.val.append(val * n)
        self.count.append(n)
        if self.max_len > 0 and len(self.val) > self.max_len:
            self.val = self.val[-self.max_len:]
            self.count = self.count[-self.max_len:]
        self.avg = sum(self.val) / sum(self.count)
        

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class F1_Score:
    __name__ = 'F1 macro'
    def __init__(self,n=3):
        self.n = n
        self.TP = np.zeros(self.n)
        self.FP = np.zeros(self.n)
        self.FN = np.zeros(self.n)

    def get_confusion_matrix(self, prediction, ground_truth, num_classes):
        gt_onehot = torch.nn.functional.one_hot(ground_truth, num_classes=num_classes).float()
        prediction = torch.argmax(prediction, dim=1)
        pd_onehot = torch.nn.functional.one_hot(prediction, num_classes=num_classes).float()
        return pd_onehot.t().matmul(gt_onehot)
    
    def __call__(self, preds, targs):
        cm = self.get_confusion_matrix(preds, targs, num_classes=3)
        print(cm)
        TP = cm.diagonal()
        FP = cm.sum(1) - TP
        FN = cm.sum(0) - TP
        self.TP += TP.float().cpu().numpy()
        self.FP += FP.float().cpu().numpy()
        self.FN += FN.float().cpu().numpy()
        
    def print(self):
        precision = self.TP / (self.TP + self.FP + 1e-12)
        recall = self.TP / (self.TP + self.FN + 1e-12)
        self.precision = precision
        self.recall = recall
        # precision = precision.mean()
        # recall = recall.mean()
        score = 2.0 * (precision * recall) / (precision + recall + 1e-12)
        score = score.mean()
        return score

    def reset(self):
        self.TP = np.zeros(self.n)
        self.FP = np.zeros(self.n)
        self.FN = np.zeros(self.n)