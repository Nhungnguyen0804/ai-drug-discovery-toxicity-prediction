import torch
import torch.nn.functional as F
import numpy as np

def masked_bce_loss(pred, target, weight=None):
    """
    Binary Cross Entropy với mask cho missing labels (NaN)
    
    Args:
        pred: [batch, num_labels] - Logits (chưa qua sigmoid)
        target: [batch, num_labels] - Labels (chứa NaN cho labels thiếu)
        weight: [num_labels] - Optional, trọng số cho từng label
    Returns:
        loss: Scalar tensor
    """
    # Mask các label KHÔNG phải NaN
    mask = ~torch.isnan(target)
    
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    
    # Thay NaN = 0 để tránh lỗi (sẽ bị mask anyway)
    target_clean = torch.where(torch.isnan(target), torch.zeros_like(target), target)
    
    # Compute loss element-wise (chưa reduce)
    loss = F.binary_cross_entropy_with_logits(pred, target_clean, reduction='none')
    
    # Apply weight TRƯỚC KHI mask (nếu có)
    if weight is not None:
        weight_expanded = weight.unsqueeze(0).expand_as(loss)
        loss = loss * weight_expanded
    
    # Apply mask và tính mean
    masked_loss = loss[mask]
    
    return masked_loss.mean()


