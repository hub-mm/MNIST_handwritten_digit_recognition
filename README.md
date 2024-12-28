# MNIST Handwritten Digit Recognition

This repository contains a simple pipeline for training a neural network on the classic MNIST dataset.  
It demonstrates:
- Downloading and preprocessing the data
- Visualising the training/test images (optional)
- Building and training a neural network
- Evaluating a trained model
- Saving and reloading the trained model for interference

## Table of Contents
1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Examples](#examples)
7. [Contributing](#contributing)

## Project Structure
<details>
<summary>Project Structure</summary>

```bash
    .
    ├── README.md
    ├── data
    │   └── MNIST
    │       └── mnist.npz
    ├── main.py
    ├── models
    │   ├── __init__.py
    │   ├── create_model.py
    │   └── saved_models
    │       ├── __init__.py
    │       └── mnist_model.keras
    ├── requirements.txt
    └── scripts
        ├── download_data.py
        ├── evaluate.py
        ├── train.py
        └── visualise.py
```

</details>

## Requirements
- **Python 3.7+**
### Packages
- **TensorFlow** (tested with TensorFlow 2.12+)
- **NumPy**
- **Matplotlib**
- **argparse** (bundled with standard Python)

*All required packages are listed in **requirements.txt***

## Installation
1. **Clone the repository:**
    ```bash
    git clone 
    ```

2. **Create Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # on macOS/Linux
    .\venv\Scripts\activate  # on Windows
    ```

3. **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Usage
- Run **main.py** script to train and evaluate the model:
    ```bash
    python main.py --epochs 3 --visualise
    ```

    #### Command Line Arguments:
    | Argument       |                  Description                  | Default      |
    |:---------------|:---------------------------------------------:|--------------|
    | --epochs       |      Number of epochs to train the model      | 3            |
    | --visualise    | If set, shows a sample image from the dataset | (flag)       |
    | --dataset_path |     Path to store/load the MNIST dataset      | ./data/MNIST |

### Examples
1. **Train for 5 epochs and visualise a sample:**
    ```bash
    python main.py --epochs 5 --visualise
    ```

2. **Train for 10 epochs without visualising:**
    ```bash
    python main.py --epochs 10
    ```

3. **Specify a custom dataset path:**
    ```bash
    python main.py --epochs 2 --dataset_path /tmp/mnist
    ```

4. **Use Default**:
    ```bash
    python main.py
    ```

#### When using the default parameters, this will:
1. Download the MNIST dataset to ./data/MNIST/ if not already present
2. Train a simple neural network for 3 epochs
3. Evaluate on the test set (10k images)
4. Save the model the ./models/saved_models/mnist_model.keras
5. Reload the model, compile it, predict the first test image, and display the predicted label vs. the true label

## Contributing
1. Fork the repository
2. Create a feature branch:
    ```bash
    git checkout -b feature/my_new_feature
    ```

3. Commit your changes:
    ```bash
    git commit -m 'Add some feature'
    ```

4. Push to the branch:
    ```bash
    git push origin feature/my_new_feature
    ```

5. Open a pull request describing what changed and why

###

**Thank you for taking the time to check this work out!**