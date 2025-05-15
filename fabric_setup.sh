#!/bin/bash

# Update and install system dependencies
echo "Updating system..."
sudo apt update
sudo apt upgrade -y

# Install Python development tools
echo "Installing Python dev tools..."
sudo apt install -y python3-dev python3-pip

# Install NVIDIA drivers
echo "Installing NVIDIA drivers..."
sudo apt install -y nvidia-driver-570
echo "NVIDIA drivers installed. A reboot will be required."


# Install Python packages
echo "Installing Python packages..."
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
echo "Verifying GPU setup..."
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"

# Run the setup script
echo "Running EEG setup script..."
python setup_eeg_project.py

echo "Setup complete!" 
echo "Run "sudo reboot" 