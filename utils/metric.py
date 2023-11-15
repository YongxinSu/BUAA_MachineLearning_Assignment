# The tensor version metric.py for saliency prediction

# You can call the calc_cc_score/KLD function by 'import metric' first in another .py.
# Then call the function by
# 'cc_score = metric.calc_cc_score(gtMAP, prMAP) '
# (Just for an instance) to obtain the cc score of each test image.


import math
import numpy as np
import torch

def KLD(q, p):
    # q: Ground-truth saliency map
    # p: Predicted saliency map
    # b = q.shape[0]
    p = normalize(p, method='sum')
    q = normalize(q, method='sum')
    return np.sum(np.where(p != 0, p * np.log((p+EPSILON) / (q+EPSILON)), 0)) 


def normalize(x, method='standard', axis=None):

    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def calc_cc_score_GPU(gtsAnns, resAnns):
    # gtsAnn: Ground-truth saliency map
    # resAnn: Predicted saliency map
    BatchSize = gtsAnns.shape[0]
    cc = torch.tensor(float(0)).cuda()
    for i in range(BatchSize):
        gtsAnn = gtsAnns[i]
        resAnn = resAnns[i]

        fixationMap = gtsAnn - torch.mean(gtsAnn)
        fixationMap = torch.reshape(fixationMap, [-1])
        if torch.max(fixationMap) > 0:
            fixationMap = fixationMap / torch.std(fixationMap)

        salMap = resAnn - torch.mean(resAnn)
        salMap = torch.reshape(salMap, [-1])
        if torch.max(salMap) > 0:
            salMap = salMap / torch.std(salMap)

        tmp_map = torch.stack((salMap, fixationMap))
        tmp_cc = torch.corrcoef(tmp_map)[0][1]
        cc = cc + tmp_cc

    return torch.div(cc, BatchSize)

def calc_cc_score(gtsAnn, resAnn):
    # gtsAnn: Ground-truth saliency map
    # resAnn: Predicted saliency map
    # b = gtsAnn.shape[0]
    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]


EPSILON = np.finfo('float').eps
EPSILON_gpu = torch.tensor(EPSILON)

def KLD_gpu(q, p):

    p = normalize_gpu(p)
    q = normalize_gpu(q)

    kl = torch.sum(torch.where(p != torch.tensor(0), p * torch.log((p+EPSILON_gpu) / (q+EPSILON_gpu)), torch.tensor(float(0)).cuda()))

    return kl


def normalize_gpu(x):

    res = x / (torch.sum(x))

    return res

if __name__ == "__main__":
    
    # test same
    x = torch.rand(size=(1, 10), requires_grad=False).float().cuda()
    y = torch.rand(size=(1, 10), requires_grad=False).float().cuda()
    
    print(KLD_gpu(x, y))
    
    x = x.cpu().contiguous().numpy()
    y = y.cpu().contiguous().numpy()
    
    print(KLD(x, y))
    
    
    