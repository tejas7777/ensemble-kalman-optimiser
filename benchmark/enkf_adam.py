import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimiser.enkf import EnKF
from model.dnn import DNN
from data.dataloader.regression_loader import OscillatoryDataLoader
from data.dataloader.classification_loader import ClassificationDataLoader
from train.batch_trainer.enkf_train import BatchTrainer as EnKFTrainer
from train.batch_trainer.enkf_adaptive_train import BatchTrainer as EnkFAdaptiveTrainer
from adam_train.batch_trainer.regression import BatchTrainer as AdamTrainer
from train.batch_trainer.enkf_iterative_train import ModelTrainer as DeepEKITrainer
import os
from datetime import datetime
from visualisation.plotter import Plotter
from data.dataloader.diabetes_data_loader import DiabetesDataLoader

def benchmark(trainer_class, model, dataset_loader, num_epochs=100, params=None, require_tensor=False):

    trainer = trainer_class(model=model, **params) if params else trainer_class(model=model)
    trainer.load_data(dataset_loader)

    trainer.train(num_epochs=num_epochs, is_plot_graph=0)
    trainer.evaluate_test()
    return trainer.train_loss, trainer.val_loss

if __name__ == '__main__':
    #dataset_loader = OscillatoryDataLoader(batch_size=10000, val_size=0.2)
    dataset_loader = DiabetesDataLoader(batch_size=100000000, val_size=0.2)

    # Determine the input and output size from the feature dimensions
    input_size = dataset_loader.train_dataset.X.shape[1]
    output_size = dataset_loader.train_dataset.y.shape[1]

    # Define the models
    model_enkf = DNN(input_size=input_size, output_size=output_size)
    model_adam = DNN(input_size=input_size, output_size=output_size)
    model_deepeki = DNN(input_size=input_size, output_size=output_size)

    # Benchmark EnKFTrainer
    enkf_train_loss, enkf_val_loss = benchmark(
        trainer_class=EnKFTrainer,
        model=model_enkf,
        dataset_loader=dataset_loader,
        num_epochs=100,
        params={"k":100, "loss_type":"mse"},
    )

    # # Benchmark EnKF Adaptive Trainer
    # enkf_adaptive_train_loss, enkf_adaptive_val_loss = benchmark(
    #     trainer_class=EnkFAdaptiveTrainer,
    #     model=model_enkf,
    #     dataset_loader=dataset_loader,
    #     num_epochs=500,
    #     params={"k":100}
    # )

    # Benchmark BatchTrainer (using Adam optimizer)
    adam_train_loss, adam_val_loss = benchmark(
        trainer_class=AdamTrainer,
        model=model_adam,
        dataset_loader=dataset_loader,
        num_epochs=100,
    )

    # Benchmark ModelTrainer (EnKF Iterative)
    enkf_iterative_train_loss, enkf_iterative_val_loss = benchmark(
    trainer_class=DeepEKITrainer,
    model=model_deepeki,
    dataset_loader=dataset_loader,
    num_epochs=100,
    require_tensor=True  # Set this to true to use tensors
)

# Initialize Plotter
plotter = Plotter()

# Prepare data for comparison plotting
loss_data = [
    (enkf_val_loss, 'EnKF Validation Loss'),
    # (enkf_adaptive_val_loss, 'EnKF (Adaptive) Validation Loss'),
    (adam_val_loss, 'Adam Validation Loss'),
    (enkf_iterative_val_loss, 'DeepEKI Validation Loss'),
]

# Plot comparison graph
plotter.plot_benchmark_graph(
    loss_data=loss_data,
    log_scale=True,
    xlabel='Epochs',
    ylabel='Loss',
    # title='Validation Loss Comparison'
)
