
# import torch
# import torchvision
# from torch.utils.data import DataLoader

import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]





def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    #print(input.size())
    #print(target.size())
    assert input.size() == target.size()
    fn = dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


# def GDLoss(x, y):
#     tp = torch.sum(x * y, dim=(0,2,3))
#     fp = torch.sum(x*(1-y),dim=(0,2,3))
#     fn = torch.sum((1-x)*y,dim=(0,2,3))
#     nominator = 2*tp + 1e-05
#     denominator = 2*tp + fp + fn + 1e-05
#     dice_score = -(nominator / (denominator+1e-8))[1:].mean()
#     return dice_score

