# ./scripts/train.py
import logging
import tensorflow as tf

from scripts.download_data import download_mnist_data
from models.create_model import create_mnist_model


logger = logging.getLogger(__name__)

def train_model(epochs=3, dataset_path='./data/MNIST'):
    (x_train, y_train), (x_test, y_test) = download_mnist_data(dataset_path)

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = create_mnist_model()
    logger.info('Create a new MNIST model.')

    logger.info(f"Starting training for {epochs} epochs...")
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_split=0.1,
        batch_size=64
    )
    logger.info('Training complete.')

    return model, (x_train, y_train), (x_test, y_test)