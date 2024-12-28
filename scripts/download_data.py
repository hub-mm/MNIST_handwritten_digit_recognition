# ./utils/download_data.py
import os
import logging
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)

def download_mnist_data(dataset_path='./data/MNIST'):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    file_path = os.path.join(dataset_path, 'mnist.npz')
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

    logger.info('Downloading MNIST data...')
    tf.keras.utils.get_file(
        fname='mnist.npz',
        origin=url,
        cache_dir=dataset_path,
        cache_subdir='.',
    )
    logger.info(f"Data downloaded to: {file_path}")

    with np.load(file_path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    logger.info(f"Number of training samples: {x_train.shape[0]}")
    logger.info(f"Number of test samples: {x_test.shape[0]}")

    return (x_train, y_train), (x_test, y_test)