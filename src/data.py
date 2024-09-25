import numpy as np
import torch
from typing import Tuple
import librosa
import matplotlib.pyplot as plt
import os
import requests
import gzip

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desired width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

def order_reduce_mnist(data, target, train=True):
    # Order the data according to the targets
    order = np.argsort(target)
    data = data[order]
    target = target[order]
    # Remove same number of samples from each class so that we have a balanced dataset
    if train == True:
        samples_per_class = [2620, 2603, 2613, 2598, 2621, 2614, 2616, 2571, 2590, 2554]
    else:
        samples_per_class = [380,  397,  387,  402,  379,  386,  384,  429,  410,  446]
    for i in range(10):
        idx = np.where(target == i)[0]
        idx = idx[samples_per_class[i]:]
        data = np.delete(data, idx, axis=0)
        target = np.delete(target, idx, axis=0)
    return data, target

def download_file(url, filename, chunk_size=128):
    print(f"Downloading {url} to {filename}")
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # Skip the first 16 bytes (header)
        f.read(16)
        # Read the rest of the file
        buf = f.read()
        # Convert to NumPy array
        data = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # Skip the first 8 bytes (header)
        f.read(8)
        # Read the rest of the file
        buf = f.read()
        # Convert to NumPy array
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels

def download_ImageMNIST(root='data'):
    """
    Downloads the MNIST dataset, then converts and returns it as numpy arrays.
    """ 
    # First we download the dataset
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(os.path.join(root,'train_images.gz')):
        download_file(r'https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', os.path.join(root,'train_images.gz'))
    if not os.path.exists(os.path.join(root,'train_labels.gz')):
        download_file(r'https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', os.path.join(root,'train_labels.gz'))
    if not os.path.exists(os.path.join(root,'test_images.gz')):
        download_file(r'https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', os.path.join(root,'test_images.gz'))
    if not os.path.exists(os.path.join(root,'test_labels.gz')):
        download_file(r'https://web.archive.org/web/20220331130319/https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', os.path.join(root,'test_labels.gz'))
    # Now turn the data into numpy arrays
    train_images = load_mnist_images(os.path.join(root,'train_images.gz'))
    train_labels = load_mnist_labels(os.path.join(root,'train_labels.gz'))
    test_images = load_mnist_images(os.path.join(root,'test_images.gz'))
    test_labels = load_mnist_labels(os.path.join(root,'test_labels.gz'))

    return train_images, train_labels, test_images, test_labels

def get_ImageMNIST(root='data') -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:     
    """ 
    Downloads the MNIST dataset (https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py).
    Returns the train and test set, with normalization transform applied.

    Returns:
        - train_set (torch.utils.data.Dataset): The training set
        - test_set (torch.utils.data.Dataset): The test set
    """       
    # Check if np files already exist and load them if they do
    if os.path.exists(os.path.join(root,'train_images.npy')) and os.path.exists(os.path.join(root,'train_labels.npy')) and os.path.exists(os.path.join(root,'test_images.npy')) and os.path.exists(os.path.join(root,'test_labels.npy')):
        train_images = np.load(os.path.join(root,'train_images.npy'))
        train_labels = np.load(os.path.join(root,'train_labels.npy'))
        test_images = np.load(os.path.join(root,'test_images.npy'))
        test_labels = np.load(os.path.join(root,'test_labels.npy'))
    # Otherwise download and load MNIST dataset with labels as integers (0-9)
    else:
        train_images, train_labels, test_images, test_labels = download_ImageMNIST(root=root)
    
    # Keep only 26000 training samples (equal across classes)
    train_images, train_labels = order_reduce_mnist(train_images, train_labels, train=True)
    # Keep only 4000 test samples (equal across classes)
    test_images, test_labels = order_reduce_mnist(test_images, test_labels, train=False)
    # Add channel dimension
    train_images = train_images[:,:,:, np.newaxis]
    test_images = test_images[:,:,:, np.newaxis]
    
    return train_images, train_labels, test_images, test_labels

def get_ImageMNIST_as_tensor(root='data', norm=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ 
    Essentialy the same as get_ImageMNIST, but returns the data as pytorch tensors instead of numpy arrays.

    Args:
        - root (str): The root directory to save the data
        - norm (bool): Whether to normalize the data
    Returns:
        - train_images (torch.Tensor): The training images
        - train_labels (torch.Tensor): The training labels
        - test_images (torch.Tensor): The test images
        - test_labels (torch.Tensor): The test labels
    """  
    from torchvision import transforms
    # Get the data
    train_images, train_labels, test_images, test_labels = get_ImageMNIST(root=root)

    # Apply transform to training and test set to normalize the data and convert to tensor
    if norm:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    train_images = torch.stack([transform(d) for d in train_images])
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_images = torch.stack([transform(d) for d in test_images])
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    return train_images, train_labels, test_images, test_labels

def download_AudioMNIST(viz = True, cleanup=True, root='data'):
    """
    Downloads the AudioMNIST dataset (https://github.com/soerenab/AudioMNIST).

    Args:
        - viz (bool): Whether to create plots of the audio data
        - cleanup (bool): Whether to delete the cloned directory of the original data after processing
    
    """
    # Download the dataset if it doesn't exist
    if not os.path.exists(os.path.join(root,'AudioMNIST')):
        os.system('git clone https://github.com/soerenab/AudioMNIST.git '+ os.path.join(root,'AudioMNIST'))
    # Check directory for viz exists
    if viz and not os.path.exists(os.path.join(root, 'viz','written_spoken_digits','AudioMNIST')):
        os.makedirs(os.path.join(root, 'viz', 'written_spoken_digits', 'AudioMNIST'))
    # Create numpy array of the data
    audio_data = []
    label_data = []
    for digit in range(0,10):
        rndm_speaker = np.random.randint(0,60) # Note random speaker and index for visualization purposes only
        rndm_index = np.random.randint(0,50)
        for speaker in range(1,61):
            for index in range(0,50):
                if speaker<10:
                    file = os.path.join(root, "AudioMNIST", "data", f"0{speaker}", f"{digit}_0{speaker}_{index}.wav")
                else:
                    file = os.path.join(root, "AudioMNIST", "data", f"{speaker}", f"{digit}_{speaker}_{index}.wav")
                audio, sample_rate = librosa.load(file)
                if viz and speaker==rndm_speaker and index==rndm_index:
                    plt.figure(figsize=(10, 4))
                    plt.plot(audio)
                    plt.title(f"Digit {digit} - Speaker {speaker} - Index {index}")
                    plt.savefig(os.path.join(root, "viz", 'written_spoken_digits', "AudioMNIST", f"digit_{digit}_speaker_{speaker}_{index}.png"))
                    plt.close()
                
                    os.system(f"cp {file} {os.path.join(root, 'viz','written_spoken_digits', 'AudioMNIST', f'digit_{digit}_speaker_{speaker}_{index}.wav')}")
                n_mfcc = 40
                mfcc = librosa.feature.mfcc(y=audio,sr=sample_rate, n_mfcc=n_mfcc)
                mfcc = padding(mfcc, n_mfcc, 44)

                if viz and speaker==rndm_speaker and index==rndm_index:
                    plt.figure(figsize=(10, 4))
                    plt.imshow(mfcc, aspect='auto', origin='lower')
                    plt.colorbar()
                    plt.title(f"Digit {digit} - Speaker {speaker} - Index {index}")
                    plt.savefig(os.path.join(root, "viz", 'written_spoken_digits', "AudioMNIST", f"digit_{digit}_speaker_{speaker}_{index}_mfcc.png"))
                    plt.close()

                audio_data.append(mfcc)
                label_data.append(digit)
    audio_data = np.stack(audio_data)          
    label_data = np.stack(label_data)
    np.save(os.path.join(root,'mnist_audio_data'), audio_data, allow_pickle =False)
    np.save(os.path.join(root,'mnist_audio_labels'), label_data, allow_pickle =False)
    if cleanup:
        os.system('rm -rf '+ os.path.join(root, 'AudioMNIST'))

def get_AudioMNIST(root='data', viz=True, cleanup=True, ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Checks if the AudioMNIST dataset (https://github.com/soerenab/AudioMNIST) is downloaded and processed, returns it if it is.
    If it isn't then will download it, split into a train and test set and for each; create two tensors, one for the audio and one for the labels, process the audio into MFCCs and save them as a pytorch dataset.

    Returns:
        - train_dataset (torch.utils.data.Dataset): The training dataset
        - test_dataset (torch.utils.data.Dataset): The test dataset

    """
    np.random.seed(42)
    # Check if np files exist and load them if they do
    if os.path.exists(os.path.join(root,'mnist_audio_data.npy')) and os.path.exists(os.path.join(root,'mnist_audio_labels.npy')):
        audio_data = np.load(os.path.join(root,'mnist_audio_data.npy'))
        audio_labels = np.load(os.path.join(root,'mnist_audio_labels.npy'))
    else:
        download_AudioMNIST(root=root, viz=viz, cleanup=cleanup)
        audio_data = np.load(os.path.join(root,'mnist_audio_data.npy'))
        audio_labels = np.load(os.path.join(root,'mnist_audio_labels.npy'))
    # Get random indices for shuffling the data and save the indices for reproducibility
    indices = np.random.permutation(audio_data.shape[0])
    np.save(os.path.join(root,'train_test_split_indices'), indices, allow_pickle =False)
    # Split the data into train and test sets
    train_indices = indices[:30000-4000]
    test_indices = indices[30000-4000:]
    train_data = audio_data[train_indices]
    train_labels = audio_labels[train_indices]
    test_data = audio_data[test_indices]
    test_labels = audio_labels[test_indices]
    # Add channel dimension and order the data arrays according to the labels
    sorted_train_indices = np.argsort(train_labels)
    train_data = train_data[sorted_train_indices][:,:,:, np.newaxis]
    train_labels = train_labels[sorted_train_indices]

    sorted_test_indices = np.argsort(test_labels)
    test_data = test_data[sorted_test_indices][:,:,:, np.newaxis]
    test_labels = test_labels[sorted_test_indices]
    
    return train_data, train_labels, test_data, test_labels

def get_AudioMNIST_as_tensor(root='data', viz=True, cleanup=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Essentialy the same as get_AudioMNIST, but returns the data as pytorch tensors instead of numpy arrays.

    Returns:
        - train_data (torch.Tensor): The training data
        - train_labels (torch.Tensor): The training labels
        - test_data (torch.Tensor): The test data
        - test_labels (torch.Tensor): The test labels

    """
    from torchvision.transforms import v2
    # Get the data
    train_data, train_labels, test_data, test_labels = get_AudioMNIST(root=root, viz=viz, cleanup=cleanup)
    # Convert to tensors
    trans = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    train_data = torch.stack([trans(d) for d in train_data])
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.stack([trans(d) for d in test_data])
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    return train_data, train_labels, test_data, test_labels

# TODO: Go over docstrings and update comments, args and returns