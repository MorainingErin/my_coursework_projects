The project focuses on training and evaluating RNN & LSTM on the Google Stock dataset, with ablation studies on data normalization and sequence length.

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

- `network/`: Contains files for each network model used in training (RNN, LSTM).
- `scripts/`: Shell scripts to execute various stages of training and evaluation.
- `dataset.py`: Defines the dataset class for loading and processing Google Stock data.
- `main.py`: The main entry point for running the training.
- `utils.py`: Contains utility functions used throughout the project.
- `worker.py`: Implements the main training framework and handles the model training process.

## Dataset

This project uses the Google stock dataset, which can be downloaded from [here](https://www.kaggle.com/datasets/rahulsah06/gooogle-stock-price).

After downloading, please unpack the dataset and place it under the root directory `./data`, or adjust the `data_path` in the `main.py` file accordingly.

## Training Process

To train and evaluate the models, follow these steps:

### 1. Determine the Base Learning Rate:

Run the cyclical learning rate script to find an optimal base learning rate.

```bash
./scripts/cyc_lr.sh
```

### 2. Run Base Model Training

Train the models with various parameters.

```bash
./scripts/run_train.sh
```

### 3. Evaluate the results

Run evaluation on the test set to validate final model performance.

```bash
./scripts/run_eval.sh
```
