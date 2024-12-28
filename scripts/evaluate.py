# ./scripts/evaluate.py
import logging
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)

def eval_model(model: tf.keras.Model,
               x_test: np.ndarray,
               y_test: np.ndarray):

    val_loss, val_acc = model.evaluate(x_test, y_test)

    logger.info(f"Test Loss: {val_loss}")
    logger.info(f"Test Accuracy: {val_acc}")

    return val_loss, val_acc