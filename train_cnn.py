import torch
import numpy as np
import matplotlib.pyplot as plt
from data_pipeline import load_data
from models import SimpleCNN
from loss_functions import denoise_loss_mse, denoise_loss_rmse, denoise_loss_rrmset
from tqdm import tqdm
import time
import os

# Configuration for training
NOISE_TYPE = 'EOG'       # Artifact type: 'EOG' or 'EMG'
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = .0005

# Create results directory structure
def create_results_dirs():
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/CNN', exist_ok=True)
    os.makedirs('results/FCNN', exist_ok=True)

if __name__ == "__main__":
    # Create directories
    create_results_dirs()
    
    print("Starting CNN training script...")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    print("Loading dataset...")
    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, std_vals = load_data(
        data_dir='eegdenoisedata', 
        noise_type=NOISE_TYPE, 
        train_ratio=0.8,
        combin_num=10
    )
    segment_length = EEG_train.shape[1]
    print(f"Data loaded successfully. Training data shape: {EEG_train.shape}")

    # Initialize model, optimizer, and loss function
    print("Initializing model...")
    model = SimpleCNN(segment_length).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Model initialized successfully")

    # Initialize history tracking
    train_mse_history = []
    val_mse_history = []
    grad_history = []

    # Prepare training/validation tensors
    print("Converting data to tensors...")
    train_noisy_tensor = torch.from_numpy(noiseEEG_train).to(device)
    train_clean_tensor = torch.from_numpy(EEG_train).to(device)
    val_noisy_tensor = torch.from_numpy(noiseEEG_val).to(device)
    val_clean_tensor = torch.from_numpy(EEG_val).to(device)
    print(f"Training tensor shape: {train_noisy_tensor.shape}")

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start_time = time.time()
        total_grad_norm = 0
        
        # Shuffle training data each epoch
        indices = torch.randperm(train_noisy_tensor.shape[0])
        train_noisy_tensor = train_noisy_tensor[indices]
        train_clean_tensor = train_clean_tensor[indices]
        batch_losses = {'mse': []}
        
        # Calculate number of batches
        n_batches = train_noisy_tensor.shape[0] // BATCH_SIZE
        
        # Iterate over mini-batches with progress bar
        pbar = tqdm(range(0, train_noisy_tensor.shape[0], BATCH_SIZE), 
                   desc=f'Epoch {epoch}/{EPOCHS}',
                   total=n_batches,
                   leave=True)
        
        for i in pbar:
            batch_noisy = train_noisy_tensor[i:i+BATCH_SIZE].unsqueeze(1).float()
            batch_clean = train_clean_tensor[i:i+BATCH_SIZE].float()
            optimizer.zero_grad()
            outputs = model(batch_noisy)
            
            # Compute MSE loss
            mse_loss = denoise_loss_mse(outputs, batch_clean)
            
            # Backward pass
            mse_loss.backward()
            
            # Calculate gradient norm before optimizer step
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += torch.sum(torch.square(p.grad.detach()))
            batch_grad_norm = torch.sqrt(total_norm).item()
            total_grad_norm += batch_grad_norm / n_batches
            
            optimizer.step()
            
            # Record MSE loss
            batch_losses['mse'].append(mse_loss.item())
            
        # Calculate average training MSE
        train_mse = float(np.mean(batch_losses['mse']))
            
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_in = val_noisy_tensor.unsqueeze(1).float()
            val_tgt = val_clean_tensor.float()
            val_out = model(val_in)
            val_mse = denoise_loss_mse(val_out, val_tgt).item()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Store history
        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse)
        grad_history.append(total_grad_norm)
        
        # Print metrics in requested format
        print(f"Epoch #: {epoch}/{EPOCHS}, Time taken: {epoch_time:.4f} secs,")
        print(f" Grads: mse= {total_grad_norm:.6f},")
        print(f" Losses: train_mse= {train_mse:.6f}, val_mse={val_mse:.6f}\n")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_in = torch.from_numpy(noiseEEG_test).unsqueeze(1).float().to(device)
        test_out = model(test_in).cpu().numpy()

    # Visualization
    if noiseEEG_test.shape[0] > 0:
        # Choose an example with significant noise (middle SNR range)

        # idx = 0  # example index
        # noisy_example = noiseEEG_test[idx]
        # clean_example = EEG_test[idx]
        # denoised_example = test_out[idx]

        idx = noiseEEG_test.shape[0] // 2  # Select middle sample which should have moderate SNR
        noisy_example = noiseEEG_test[idx] * std_vals[idx]  # Denormalize
        clean_example = EEG_test[idx] * std_vals[idx]  # Denormalize
        denoised_example = test_out[idx] * std_vals[idx]  # Denormalize

        # Plot training loss curves
        plt.figure(figsize=(12, 8))
        
        # Plot MSE Loss
        plt.subplot(2, 1, 1)
        plt.plot(range(1, EPOCHS+1), train_mse_history, label='Train MSE')
        plt.plot(range(1, EPOCHS+1), val_mse_history, label='Val MSE')
        plt.title('MSE Loss Progress (CNN)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot Gradient Norm
        plt.subplot(2, 1, 2)
        plt.plot(range(1, EPOCHS+1), grad_history, label='Gradient Norm')
        plt.title('Gradient Norm Progress (CNN)')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/CNN/training_progress_{NOISE_TYPE}_cnn.png')
        plt.show()

        # Plot noisy vs clean and denoised vs clean signals
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(clean_example, label='Clean EEG')
        plt.plot(noisy_example, label='Noisy EEG', alpha=0.7)
        plt.title('Noisy vs Clean Signal')
        plt.legend(loc='upper right')
        plt.subplot(2, 1, 2)
        plt.plot(clean_example, label='Clean EEG')
        plt.plot(denoised_example, label='Denoised (Model)', alpha=0.7)
        plt.title('Denoised vs Clean Signal')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'results/CNN/denoising_example_{NOISE_TYPE}_cnn.png')
        plt.show()

    # Save model weights
    torch.save(model.state_dict(), f'results/CNN/cnn_model_{NOISE_TYPE}.pth')
