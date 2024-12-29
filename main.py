# ./main.py
from pathlib import Path
import logging
import argparse
import tensorflow as tf
import numpy as np

from scripts.train import train_model
from scripts.evaluate import eval_model
from scripts.visualise import visualise_data
from web_app.app import app


def main():
    try:
        Path('./models/saved_models/mnist_model.keras').resolve(strict=True)
    except FileNotFoundError:
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

    else:
        parser = argparse.ArgumentParser(description='Run the MNIST web app')
        parser.add_argument('--host', default='127.0.0.1', help='Host to listen on')
        parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
        parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')
        args = parser.parse_args()

        app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()