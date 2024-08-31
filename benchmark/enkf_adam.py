import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimiser.enkf import EnKF
from model.dnn import DNN
#from model.dnn import DiabetesDataDNN as DNN
from data.dataloader.regression_loader import OscillatoryDataLoader
from data.dataloader.classification_loader import ClassificationDataLoader
from data.dataloader.saddle_point_dataset_loader import SaddlePointDataLoader
from data.dataloader.ackley_function_dataset_loader import AckleyFunctionDataLoader
from data.dataloader.greiwank_function_dataset_loader import GriewankFunctionDataLoader
from data.dataloader.michalewicz_function_dataset_loader import MichalewiczFunctionDataLoader
from train.batch_trainer.enkf_train import BatchTrainer as EnKFTrainer
from train.batch_trainer.enkf_adaptive_train import BatchTrainer as EnkFAdaptiveTrainer
from adam_train.batch_trainer.regression import BatchTrainer as AdamTrainer
from train.batch_trainer.enkf_iterative_train import ModelTrainer as DeepEKITrainer
from train.batch_trainer.enkf_legacy_trainer import ModelTrainer as EnKFIterativeTrainer
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

def benchmark_classification(trainer_class, model, dataset_loader, num_epochs=100, params=None, require_tensor=False):

    trainer = trainer_class(model=model, **params) if params else trainer_class(model=model)
    trainer.load_data(dataset_loader)

    trainer.train(num_epochs=num_epochs, is_plot_graph=0)
    trainer.evaluate_accuracy()
    return trainer.train_loss, trainer.val_loss

if __name__ == '__main__':
    dataset_loader = OscillatoryDataLoader(batch_size=10000, val_size=0.2)
    #dataset_loader = SaddlePointDataLoader(num_samples=10000, noise_level=0.1, val_size=0.2, test_size=0.2, batch_size=1000000)
    #dataset_loader = AckleyFunctionDataLoader(num_samples=10000, dimension=100, noise_level=0.9, batch_size=10000000)
    #dataset_loader = GriewankFunctionDataLoader(num_samples=10000, dimension=10, noise_level=0.9, batch_size=10000000)
    #dataset_loader = MichalewiczFunctionDataLoader(num_samples=10000, dimension=10, m=10, noise_level=0.1, batch_size=10000)

    #dataset_loader = DiabetesDataLoader(batch_size=100000000, val_size=0.2)
    #dataset_loader = ClassificationDataLoader(test_size=0.3, val_size=0.3, batch_size=1000)

    # Determine the input and output size from the feature dimensions
    input_size = dataset_loader.train_dataset.X.shape[1]
    output_size = dataset_loader.train_dataset.y.shape[1]
    #output_size = len(torch.unique(dataset_loader.train_dataset.y))

    # Define the models
    model_enkf = DNN(input_size=input_size, output_size=output_size)
    model_adam = DNN(input_size=input_size, output_size=output_size)
    model_deepeki = DNN(input_size=input_size, output_size=output_size)
    model_iterative = DNN(input_size=input_size, output_size=output_size)

    # Benchmark EnKFTrainer
    enkf_train_loss, enkf_val_loss = benchmark(
        trainer_class=EnKFTrainer,
        model=model_enkf,
        dataset_loader=dataset_loader,
        num_epochs=100,
        params={"k":50},
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
    deepeki_trian_loss, deep_eki_val_loss = benchmark(
    trainer_class=DeepEKITrainer,
    model=model_deepeki,
    dataset_loader=dataset_loader,
    num_epochs=100,
    require_tensor=True  # Set this to true to use tensors
    )

    #  # Benchmark ModelTrainer (EnKF Iterative)
    enkf_iterative_train_loss, enkf_iterative_val_loss = benchmark(
    trainer_class=EnKFIterativeTrainer,
    model=model_iterative,
    dataset_loader=dataset_loader,
    num_epochs=100,
    require_tensor=True  # Set this to true to use tensors
    )



# Initialize Plotter
plotter = Plotter()

# Prepare data for comparison plotting
loss_data = [
    (enkf_val_loss, 'EnKF (Modification)'),
    (deep_eki_val_loss, 'EnKF (Stuart)'),
    (adam_val_loss, 'Adam'),
    (enkf_iterative_val_loss, 'EnKF Haber (Original)'),
]

# Plot comparison graph
plotter.plot_benchmark_graph(
    loss_data=loss_data,
    log_scale=True,
    xlabel='Epochs',
    ylabel='Loss',
    title='Validation Loss Comparison'
)
