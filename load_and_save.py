import pickle  # load the binary files
import numpy as np  # matrix processing library
import matplotlib.pyplot as plt  # draw
import skimage.io as io
import os


#  function to load a single CIFAR-10 batch
def load_cifar_batch(filename):
    #  'rb': read binary file
    # "./data_batch_1"
    with open(filename, 'rb') as file:
        datadict = pickle.load(file, encoding='bytes')
        X = datadict[b'data']  # binary 'data'
        Y = datadict[b'labels']  # a = {key: value}  a[key] == value
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")  # channel, height, width  '3':rgb
        # 'transpose':hwc  'astype':convert pixel to int
        Y = np.array(Y)
        return X, Y


# function to save images
def save_images(images, labels, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, image in enumerate(images):  # enumerate: number counter from 0 by default
        io.imsave(f"{directory}/image_{i}_label_{labels[i]}.png", image)


if __name__ == '__main__':
    batch_files = ['./cifar-10-batches-py/data_batch_1', './cifar-10-batches-py/data_batch_2', './cifar-10-batches-py/data_batch_3',
                   './cifar-10-batches-py/data_batch_4', './cifar-10-batches-py/data_batch_5', './cifar-10-batches-py/test_batch']

    for batch_file in batch_files:
        X, Y = load_cifar_batch(batch_file)
        save_images(X, Y, 'cifar_images/' + batch_file)





