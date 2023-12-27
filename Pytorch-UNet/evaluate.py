import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff

criterion = criterion = nn.CrossEntropyLoss()

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    val_loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float)

            # predict the mask
            mask_pred = net(image)
            
            # if net.n_classes == 1:
            #     assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            #     mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            #     # compute the Dice score
            #     dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            # else:
            assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            # convert to one-hot format
            #mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()  #WE DONT WANT ONE HOT BECAUSE IT MAKES CLASSES MUTUALLY EXCLUSIVE 
            #mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            #dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            val_loss += criterion(mask_pred,mask_true)
            dice_score = val_loss

    net.train()
    return dice_score / max(num_val_batches, 1)



#--- actual function 
def pixel_wise_confusion_matrix(pred_mask,true_mask,):
    """
    takes in two binary arrays and overlays them to produde a 
    class array that assigns TP,TN,FN, or FP to each pixel.

    Parameters:
    path1 (str): a string of path to a binary mask.
    path2 (str): a string og path to a binary mask.  

    Returns:
    np array of classes 0: TN, 1:FP, 2:FN, 3:TP
    """

    # Load the binary masks
    for array in (pred_mask,true_mask):
        assert np.all(np.isin(array, [0, 1])), f"Arrays must contain only binary values (0 or 1)."
  
    # Ensure that both masks have the same shape
    assert pred_mask.shape == true_mask.shape, f"Arrays must be the same shape."

    # Convert to boolean arrays
    positive_gt = true_mask == 1
    positive_pred = pred_mask == 1
    negative_gt = ~positive_gt
    negative_pred = ~positive_pred

    # Calculate TP, TN, FP, FN
    TP = np.sum(positive_gt & positive_pred)
    TN = np.sum(negative_gt & negative_pred)
    FP = np.sum(negative_gt & positive_pred)
    FN = np.sum(positive_gt & negative_pred)

    # Create a single array representing TN, FP, FN, TP
    result_array = np.zeros_like(true_mask, dtype=np.int8)
    result_array[positive_gt & positive_pred] = 3  # True Positives
    result_array[negative_gt & negative_pred] = 2  # True Negatives
    result_array[negative_gt & positive_pred] = 1  # False Positives
    result_array[positive_gt & negative_pred] = 0  # False Negatives

    return result_array

# def evaluate(net, dataloader, device, amp):
#     net.eval()
#     num_val_batches = len(dataloader)
#     dice_score = 0

#     # iterate over the validation set
#     with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
#         for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#             image, mask_true = batch['image'], batch['mask']

#             # move images and labels to correct device and type
#             image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
#             mask_true = mask_true.to(device=device, dtype=torch.float)

#             # predict the mask
#             mask_pred = net(image)
#             print(mask_pred.size())

#             if net.n_classes == 1:
#                 assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
#                 mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
#                 # compute the Dice score
#                 dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
#             else:
#                 assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
#                 # convert to one-hot format
#                 mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()  #WE DONT WANT ONE HOT BECAUSE IT MAKES CLASSES MUTUALLY EXCLUSIVE 
#                 mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#                 # compute the Dice score, ignoring background
#                 dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

#     net.train()
#     return dice_score / max(num_val_batches, 1)
