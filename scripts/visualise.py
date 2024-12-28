# ./scripts/visualise.py
import logging
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

def visualise_data(x_test, y_test, index=0):
    logger.info(f"Visualising image at index: {index}")

    plt.imshow(x_test[0], cmap='binary')
    plt.title(f"Label: {y_test[index]}")
    plt.colorbar()
    plt.show()