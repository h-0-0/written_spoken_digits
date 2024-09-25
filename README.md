# written_spoken_digits
Toolkit for downloading and creating the multimodal written [MNIST](https://web.archive.org/web/20220331130319/http://yann.lecun.com/exdb/mnist/) and spoken [AudioMNIST](https://github.com/soerenab/AudioMNIST) digits dataset.

## Using the repo
'data.py' includes code for downloading both the image and audio datasets as well as creating the joint multimodal dataset. 
```{python}
from data import get_ImageMNIST, get_AudioMNIST
train_images, train_labels, test_images, test_labels = get_ImageMNIST()
train_audio, train_labels, test_audio, test_labels = get_AudioMNIST()
```
Note that:
- The labels returned from both are the same.
- Each item returned is a numpy array.
- Arrays are ordered such that i-th entry for all training arrays is a sample, and same for testing.
If you would like back tensors instead then try:
```{python}
from data import get_ImageMNIST_as_tensor, get_AudioMNIST_as_tensor
train_images, train_labels, test_images, test_labels = get_ImageMNIST_as_tensor(norm=True) # If norm true will also normalize
train_audio, train_labels, test_audio, test_labels = get_AudioMNIST_as_tensor()
```
If you would like to use a pytorch dataloader:
```{python}
from data import get_torch_data_loaders
train_loader, test_loader = get_torch_data_loaders(batch_size, norm=True)
```
If you would like to create some visualizations simply run in your terminal:
```{bash}
python src/viz.py
```
And/or set 'viz=True' as a keyword argument when using select functions.

Currently this is mainly aimed at loading in the dataset as numpy arrays and/or as tensors/pytorch dataloader. I'm open for support for other frameworks being added, so please raise an issue and consider contributing if you think any additions would be useful. Make sure to @ me so it gets my attention. 

## Information on the dataset
Some notes on the data itself:
- As Audio MNIST has many fewer samples we discard many of the image samples.
- We use 26,000 samples for training and 4,000 for testing.
- For each split we match for each sample in Audio MNIST for each class to a sample of the corresponding class in the same split from Image MNIST.
- In terms of preprocessing the audio data we extract the Mel-Frequency Cepstral Coefficients (MFCC), using 40 as the number of coefficients and padding the time steps out to 44. 

## Extras
To help avoid package dependency issue I only import libraries where and when required, please refer to the requirements.txt to see which packages you will need to use each part of the repo. 

If there is an error during downloading the Audio and/or Image datasets it's probably due to some problem with the servers where the datasets are stored. Wait a few hours (or a day) and then try again. If the issue persists there may be some other problem at hand, in which case, please submit an issue. 

Finally, please consider ⭐️ing this repo if you find it useful, so others can find it more easily.