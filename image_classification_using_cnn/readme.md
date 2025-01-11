The project focuses on training and evaluating various CNN models (LeNet, VGG, ResNet) on the CIFAR-10 dataset, with ablation studies on data augmentation and dropout.

## Requirements

To run this code, ensure the following packages and versions are installed:

- Python 3.10
- PyTorch 2.4.1
- CUDA 12.4

Install other required packages by running:
```
pip install -r requirements.txt
```

## Code Structure

- `network/`: Contains files for each network model used in training (LeNet, VGG, ResNet).
- `scripts/`: Shell scripts to execute various stages of training and evaluation.
- `dataset.py`: Defines the dataset class for loading and processing CIFAR-10 data.
- `main.py`: The main entry point for running the training.
- `utils.py`: Contains utility functions used throughout the project.
- `worker.py`: Implements the main training framework and handles the model training process.

## Dataset

This project uses the CIFAR-10 dataset, a widely-used benchmark dataset in image classification, which can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

After downloading, please unpack the dataset and place it under the root directory `./cifar-10-batches-py`, or adjust the `data_path` in the `main.py` file accordingly.

## Training Process

To train and evaluate the models, follow these steps:

### 1. Determine the Base Learning Rate:

Run the cyclical learning rate script to find an optimal base learning rate.

```bash
./cyc_lr.sh
```

### 2. Run Base Model Training

Train the base models without any additional improvements.

```bash
./run_base.sh
```

### 3. Run Ablation Study for Data Augmentation and Dropout

Conduct ablation studies to evaluate the effects of data augmentation and dropout.

```bash
./run_dataaug.sh
./run_dropout.sh
```

### 4. Run Improved Model Training

Train the improved models with selected enhancements.

```bash
./run_imp.sh
```

### 5. Evaluate on Test Set

Run evaluation on the test set to validate final model performance.

```bash
./run_eval.sh
```
