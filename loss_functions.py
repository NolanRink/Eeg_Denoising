import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal

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

def calculate_temporal_rrmse(y_hat, x):
    """
    Calculate temporal RRMSE = RMS(y_hat – x)/RMS(x)
    """
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        
    rms_diff = np.sqrt(np.mean((y_hat - x) ** 2))
    rms_x = np.sqrt(np.mean(x ** 2))
    return rms_diff / rms_x if rms_x != 0 else rms_diff

def calculate_spectral_rrmse(y_hat, x, fs=250):
    """
    Calculate spectral RRMSE = RMS(PSD(y_hat) – PSD(x)) / RMS(PSD(x))
    """
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        
    # Calculate PSD using Welch's method
    f_y_hat, psd_y_hat = signal.welch(y_hat, fs=fs)
    f_x, psd_x = signal.welch(x, fs=fs)
    
    rms_diff = np.sqrt(np.mean((psd_y_hat - psd_x) ** 2))
    rms_x = np.sqrt(np.mean(psd_x ** 2))
    return rms_diff / rms_x if rms_x != 0 else rms_diff

def calculate_corrcoef(y_hat, x):
    """
    Calculate correlation coefficient between denoised and clean signals
    """
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        
    # flatten everything
    y = y_hat.reshape(-1)
    x = x.reshape(-1)
    # subtract means
    y = y - y.mean()
    x = x - x.mean()
    # covariance and standard deviations
    cov = np.mean(y * x)
    return cov / (np.std(y) * np.std(x))

def calculate_final_metrics(denoised, clean, fs=250):
    """
    Calculate final metrics (ACC, RRMSE_t, RRMSE_s)
    """
    rrmse_t = calculate_temporal_rrmse(denoised, clean)
    rrmse_s = calculate_spectral_rrmse(denoised, clean, fs)
    acc = calculate_corrcoef(denoised, clean)
    return {
        'RRMSE_t': float(rrmse_t),
        'RRMSE_s': float(rrmse_s),
        'ACC': float(acc),
    } 