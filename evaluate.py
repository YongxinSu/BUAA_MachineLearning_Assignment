import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metric import calc_cc_score, KLD
from utils.dice_score import multiclass_dice_coeff, dice_coeff
import math
import numpy as np


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    net.mode = 'infer'
    num_val_batches = len(dataloader)
    dice_score = 0
    kld_score = 0
    cc_score = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['label']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            # predict the mask
            mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            # mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            # compute the Dice score
            # print(mask_pred.shape, mask_true.shape)
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            
            # calc KDL
            
            mask_pred = mask_pred.detach().cpu().numpy().reshape(-1)
            mask_true = mask_true.detach().cpu().numpy().reshape(-1)
            
            t = KLD(mask_pred, mask_true)
            if not math.isnan(t):
                kld_score =  kld_score + t
            
            # calc cc
            t = calc_cc_score(mask_pred, mask_true)
            if not math.isnan(t):
                cc_score = cc_score + t
            
    dice_score = dice_score / max(num_val_batches, 1)
    kld_score = kld_score / max(num_val_batches, 1)
    cc_score = cc_score / max(num_val_batches, 1)
    
    net.mode = 'train'
    net.train()
    return {'dice': dice_score, 'kld': kld_score, 'cc': cc_score}

 

    
    
