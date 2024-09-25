import os
import torch
from data import get_AudioMNIST_as_tensor, get_ImageMNIST_as_tensor

def get_torch_data_loaders(batch_size, root='data', cleanup=True, viz=False, norm=True):
    """
    Returns the train and test data loaders for the bi-modal dataset consisting of spoken mnist (audio) and written mnist (images).
    The data is split into a train and test set, with each set containing 30000 and 4000 samples respectfully.

    Args:
        - batch_size (int): The batch size

    Returns:
        - train_loader (torch.utils.data.DataLoader): The training data loader
        - test_loader (torch.utils.data.DataLoader): The test data loader

    """
    # Check if datasets exist and load them if they do
    if os.path.exists(os.path.join('data','digits_train_data.pt')) and os.path.exists(os.path.join('data','digits_test_data.pt')):
        train_data = torch.load('data/digits_train_data.pt')
        test_data = torch.load('data/digits_test_data.pt')
    else:
        audio_train_data, audio_train_labels, audio_test_data, audio_test_labels = get_AudioMNIST_as_tensor(root='data', viz=True, cleanup=True)
        images_train_data, images_train_labels, images_test_data, images_test_labels = get_ImageMNIST_as_tensor(root='data', norm=True)
        assert (audio_train_labels == images_train_labels).all()
        assert (audio_test_labels == images_test_labels).all()
        # Combine training sets so that we match images and audio samples according to label
        print(f"Audio train data shape: {audio_train_labels.shape}")
        train_data = torch.utils.data.TensorDataset(images_train_data, audio_train_data, images_train_labels)
        # Combine test sets so that we match images and audio samples according to label
        test_data = torch.utils.data.TensorDataset(images_test_data, audio_test_data, images_test_labels)
        # Save the datasets
        torch.save(train_data, 'data/digits_train_data.pt')
        torch.save(test_data, 'data/digits_test_data.pt')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader