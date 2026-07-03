"""Validation and pixel-wise confusion matrix.

NOTE (behavior preserved intentionally): ``evaluate`` returns the mean
CrossEntropy loss over the validation set. The name ``dice_score`` is kept from
the original code — the Dice computation was commented out upstream and is left
that way so training numerics are identical to before. Do not "fix" this here.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

criterion = nn.CrossEntropyLoss()


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

            # NOTE: masks are multi-label (classes not mutually exclusive), so
            # one-hot Dice is intentionally not used; loss is CrossEntropy.
            assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            val_loss += criterion(mask_pred, mask_true)
            dice_score = val_loss

    net.train()
    return dice_score / max(num_val_batches, 1)


def pixel_wise_confusion_matrix(pred_mask, true_mask):
    """Overlay two binary arrays into a per-pixel class array.

    Returns an int8 array encoded as 0:FN, 1:FP, 2:TN, 3:TP.
    """
    # Load the binary masks
    for array in (pred_mask, true_mask):
        assert np.all(np.isin(array, [0, 1])), "Arrays must contain only binary values (0 or 1)."

    # Ensure that both masks have the same shape
    assert pred_mask.shape == true_mask.shape, "Arrays must be the same shape."

    # Convert to boolean arrays
    positive_gt = true_mask == 1
    positive_pred = pred_mask == 1
    negative_gt = ~positive_gt
    negative_pred = ~positive_pred

    # Create a single array representing FN, FP, TN, TP
    result_array = np.zeros_like(true_mask, dtype=np.int8)
    result_array[positive_gt & positive_pred] = 3  # True Positives
    result_array[negative_gt & negative_pred] = 2  # True Negatives
    result_array[negative_gt & positive_pred] = 1  # False Positives
    result_array[positive_gt & negative_pred] = 0  # False Negatives

    return result_array
