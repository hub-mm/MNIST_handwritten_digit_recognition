# ./main.py
import logging
import argparse
import tensorflow as tf
import numpy as np

from scripts.train import train_model
from scripts.evaluate import eval_model
from scripts.visualise import visualise_data


def main():
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='MNIST training and evaluation')
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of epochs'
    )
    parser.add_argument(
        '--visualise',
        action='store_true',
        help='If set, visualise a sample image'
    )
    parser.add_argument(
        '--dataset-path',
        type=str, default='./data/MNIST',
        help='Path to store the MNIST dataset'
    )
    args = parser.parse_args()

    model, (x_train, y_train), (x_test, y_test) = train_model(
        epochs=args.epochs,
        dataset_path=args.dataset_path
    )

    if args.visualise:
        visualise_data(x_test, y_test, index=0)

    eval_model(model, x_test, y_test)

    save_path = './models/saved_models/mnist_model.keras'
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")

    try:
        reload_model = tf.keras.models.load_model(save_path)
        reload_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        prediction = reload_model.predict(x_test[:1])
        predicted_label = np.argmax(prediction[0])
        actual_label = y_test[0]

        print(f"Model prediction: {predicted_label}")
        print(f"True label: {actual_label}")

        logger.info(f"Predictions for the first sample: {prediction}")
    except Exception as e:
        logger.error(f"Could not reload the model: {e}")

if __name__ == '__main__':
    main()