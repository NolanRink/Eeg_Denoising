import os
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt

def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))

def random_signal(signal, combin_num):
    random_result = []
    for i in range(combin_num):
        random_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0], signal.shape[1])
        random_result.append(shuffled_dataset)
    return np.array(random_result)

def load_data(data_dir='eegdenoisedata', noise_type='EOG', train_ratio=0.8, combin_num=10):
    """
    Load and prepare EEG and artifact data for denoising.
    
    Args:
        data_dir: Directory containing the data files
        noise_type: 'EOG' or 'EMG'
        train_ratio: Fraction of data to use for training
        combin_num: Number of random combinations to generate
    """
    # Load data files
    eeg_file = os.path.join(data_dir, "EEG_all_epochs.npy")
    artifact_file = os.path.join(data_dir, f"{noise_type}_all_epochs.npy")
    
    if not os.path.isfile(eeg_file) or not os.path.isfile(artifact_file):
        raise FileNotFoundError(f"Data files not found in {data_dir}")
        
    EEG_all = np.load(eeg_file)
    noise_all = np.load(artifact_file)
    
    # Ensure same segment length
    seg_len = min(EEG_all.shape[1], noise_all.shape[1])
    EEG_all = EEG_all[:, :seg_len]
    noise_all = noise_all[:, :seg_len]
    
    return prepare_data(EEG_all, noise_all, combin_num, train_ratio, noise_type)

def prepare_data(EEG_all, noise_all, combin_num, train_per, noise_type):
    # Random permutation of signals
    EEG_all_random = np.squeeze(random_signal(signal=EEG_all, combin_num=1))
    noise_all_random = np.squeeze(random_signal(signal=noise_all, combin_num=1))

    # Match segment counts based on noise type
    if noise_type == 'EMG':
        reuse_num = noise_all_random.shape[0] - EEG_all_random.shape[0]
        EEG_reuse = EEG_all_random[0:reuse_num, :]
        EEG_all_random = np.vstack([EEG_reuse, EEG_all_random])
    elif noise_type == 'EOG':
        EEG_all_random = EEG_all_random[0:noise_all_random.shape[0]]

    # Split data
    timepoint = noise_all_random.shape[1]
    train_num = round(train_per * EEG_all_random.shape[0])
    validation_num = round((EEG_all_random.shape[0] - train_num) / 2)

    # Training data
    train_eeg = EEG_all_random[0:train_num, :]
    train_noise = noise_all_random[0:train_num, :]
    
    # Validation data
    validation_eeg = EEG_all_random[train_num:train_num + validation_num, :]
    validation_noise = noise_all_random[train_num:train_num + validation_num, :]
    
    # Test data
    test_eeg = EEG_all_random[train_num + validation_num:, :]
    test_noise = noise_all_random[train_num + validation_num:, :]

    # Generate augmented training data
    EEG_train = random_signal(signal=train_eeg, combin_num=combin_num).reshape(combin_num * train_eeg.shape[0], timepoint)
    NOISE_train = random_signal(signal=train_noise, combin_num=combin_num).reshape(combin_num * train_noise.shape[0], timepoint)

    # Training set noise injection
    SNR_train_dB = np.random.uniform(-7, 2, (EEG_train.shape[0]))
    SNR_train = 10 ** (0.1 * SNR_train_dB)

    noiseEEG_train = []
    for i in range(EEG_train.shape[0]):
        eeg = EEG_train[i]
        noise = NOISE_train[i]
        coe = get_rms(eeg) / (get_rms(noise) * SNR_train[i])
        noise = noise * coe
        noiseEEG_train.append(noise + eeg)

    noiseEEG_train = np.array(noiseEEG_train)

    # Standardize training data
    EEG_train_end_standard = []
    noiseEEG_train_end_standard = []
    for i in range(noiseEEG_train.shape[0]):
        std_value = np.std(noiseEEG_train[i])
        EEG_train_end_standard.append(EEG_train[i] / std_value)
        noiseEEG_train_end_standard.append(noiseEEG_train[i] / std_value)

    noiseEEG_train_end_standard = np.array(noiseEEG_train_end_standard)
    EEG_train_end_standard = np.array(EEG_train_end_standard)

    # Validation set preparation
    SNR_val_dB = np.linspace(-7.0, 2.0, num=10)
    SNR_val = 10 ** (0.1 * SNR_val_dB)

    EEG_val = []
    noise_EEG_val = []
    for i in range(10):
        for j in range(validation_eeg.shape[0]):
            eeg = validation_eeg[j]
            noise = validation_noise[j]
            coe = get_rms(eeg) / (get_rms(noise) * SNR_val[i])
            noise = noise * coe
            noise_EEG_val.append(noise + eeg)
            EEG_val.append(eeg)

    noise_EEG_val = np.array(noise_EEG_val)
    EEG_val = np.array(EEG_val)

    # Standardize validation data
    EEG_val_end_standard = []
    noiseEEG_val_end_standard = []
    for i in range(noise_EEG_val.shape[0]):
        std_value = np.std(noise_EEG_val[i])
        EEG_val_end_standard.append(EEG_val[i] / std_value)
        noiseEEG_val_end_standard.append(noise_EEG_val[i] / std_value)

    noiseEEG_val_end_standard = np.array(noiseEEG_val_end_standard)
    EEG_val_end_standard = np.array(EEG_val_end_standard)

    # Test set preparation
    SNR_test_dB = np.linspace(-7.0, 2.0, num=10)
    SNR_test = 10 ** (0.1 * SNR_test_dB)

    EEG_test = []
    noise_EEG_test = []
    for i in range(10):
        for j in range(test_eeg.shape[0]):
            eeg = test_eeg[j]
            noise = test_noise[j]
            coe = get_rms(eeg) / (get_rms(noise) * SNR_test[i])
            noise = noise * coe
            noise_EEG_test.append(noise + eeg)
            EEG_test.append(eeg)

    noise_EEG_test = np.array(noise_EEG_test)
    EEG_test = np.array(EEG_test)

    # Standardize test data
    EEG_test_end_standard = []
    noiseEEG_test_end_standard = []
    std_VALUE = []
    for i in range(noise_EEG_test.shape[0]):
        std_value = np.std(noise_EEG_test[i])
        std_VALUE.append(std_value)
        EEG_test_end_standard.append(EEG_test[i] / std_value)
        noiseEEG_test_end_standard.append(noise_EEG_test[i] / std_value)

    std_VALUE = np.array(std_VALUE)
    noiseEEG_test_end_standard = np.array(noiseEEG_test_end_standard)
    EEG_test_end_standard = np.array(EEG_test_end_standard)

    return (noiseEEG_train_end_standard, EEG_train_end_standard,
            noiseEEG_val_end_standard, EEG_val_end_standard,
            noiseEEG_test_end_standard, EEG_test_end_standard,
            std_VALUE)

# Create results directory structure
def create_results_dirs():
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/CNN', exist_ok=True)
    os.makedirs('results/FCNN', exist_ok=True)

if __name__ == "__main__":
    # Create directories
    create_results_dirs()
    
    # Test data loading
    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, std_vals = load_data(
        data_dir='eegdenoisedata',
        noise_type='EOG',
        train_ratio=0.7,
        combin_num=10
    )
    
    # Print shapes
    print("\nData Shapes:")
    print(f"Training: {EEG_train.shape}")
    print(f"Validation: {EEG_val.shape}")
    print(f"Test: {EEG_test.shape}")
    
    # Print first 5 elements of clean EEG training data
    print("\nFirst 5 elements of clean EEG training data:")
    print(EEG_train[0][:5])
    
    # Plot sample EEG data
    plt.figure(figsize=(12, 4))
    plt.plot(EEG_train[0], label='Clean EEG')
    plt.plot(noiseEEG_train[0], label='Noisy EEG', alpha=0.7)
    plt.title('Sample EEG Signal')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/data/sample_eeg_signal.png')
    plt.show()
    
    # Plot multiple samples to show variability
    plt.figure(figsize=(15, 12))
    
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(EEG_train[i], alpha=0.7, label='Clean EEG')
        plt.plot(noiseEEG_train[i], alpha=0.7, label='Noisy EEG')
        plt.title(f'EEG Sample {i+1}: Clean vs Noisy')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/data/paired_eeg_samples.png')
    plt.show()
