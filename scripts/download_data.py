# ./scripts/download_data.py
import tensorflow as tf
import os
import numpy as np


def download_mnist_data(dataset_path='./data/MNIST'):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    file_path = os.path.join(dataset_path, 'mnist.npz')
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

    # Download the dataset using tf.keras.utils.get_file
    tf.keras.utils.get_file(
        fname='mnist.npz',
        origin=url,
        cache_dir=dataset_path,
        cache_subdir='.',
    )

    # Load the dataset from the downloaded file
    with np.load(file_path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    print('Data downloaded to:', dataset_path)
    print(f"Number of training samples: {x_train.shape[0]}")
    print(f"Number of training labels: {y_train.shape[0]}")
    print(f"Number of test samples: {x_test.shape[0]}")
    print(f"Number of test labels: {y_test.shape[0]}")

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    download_mnist_data()