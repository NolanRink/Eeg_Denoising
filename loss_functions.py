import torch
import torch.nn.functional as F

def denoise_loss_mse(denoise, clean):
    """
    Mean Squared Error (MSE) loss between denoised signal and clean signal.
    """
    # Use element-wise MSE and then average
    return F.mse_loss(denoise, clean, reduction='mean')

def denoise_loss_rmse(denoise, clean):
    """
    Root Mean Squared Error (RMSE) between denoised signal and clean signal.
    """
    mse_val = F.mse_loss(denoise, clean, reduction='mean')
    return torch.sqrt(mse_val)

def denoise_loss_rrmset(denoise, clean):
    """
    Relative RMSE: ratio of RMSE(denoise, clean) to RMSE(clean, zero).
    """
    rmse1 = denoise_loss_rmse(denoise, clean)
    # Compute RMSE of clean signal relative to zero signal (baseline)
    rmse2 = denoise_loss_rmse(clean, torch.zeros_like(clean))
    return rmse1 / rmse2 if rmse2.item() != 0 else rmse1 