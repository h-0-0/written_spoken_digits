from loader import get_torch_data_loaders
import matplotlib.pyplot as plt
import os
import argparse


if __name__ == '__main__':
    """
    Following code is for testing the data fetching and pre-processing functions, simply run this file to test them.
    Produces some visualizations of the data and saves some of the audio in listenable format.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='viz', help='The root directory to save the data')
    args = parser.parse_args()
    root = args.root
    print("------ Find visualisations in the {root} directory ------")
    n_b = 4
    train_loader, test_loader = get_torch_data_loaders(4)
    image_batch, audio_batch, label_batch = next(iter(train_loader))
    print(f"Image batch shape: {image_batch.shape}")
    print(f"Audio batch shape: {audio_batch.shape}")
    print(f"Label batch shape: {label_batch.shape}")
    # Create plot of subplots where each column shows the image and mfcc audio with title telling us the label for one sample from the batch
    fig, axes = plt.subplots(2, n_b, figsize=(10, 5))
    for i in range(n_b):
        axes[0, i].imshow(image_batch[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f"Label: {label_batch[i]}")
        axes[1, i].imshow(audio_batch[i].squeeze(), cmap='gray')
    if not os.path.exists('viz/written_spoken_digits'):
        os.makedirs('viz/written_spoken_digits')
    plt.savefig('viz/written_spoken_digits/batch.png')
    # Plot bar chart of number of samples in each class
    # plt.figure()
    # plt.bar(range(10), [len(train_loader.dataset[i]) for i in range(10)])
    # plt.xlabel('Class')
    # plt.ylabel('Number of samples')
    # plt.title('Number of samples in each class')
    # plt.xticks(range(10))
    # plt.savefig('viz/class_dist.png')