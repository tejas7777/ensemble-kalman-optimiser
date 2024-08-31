import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimiser.enkf import EnKF
from model.dnn import DNNClassification as DNN 
from data.dataloader.iris_dataloader import IrisDataLoader
from data.dataloader.classification_loader import ClassificationDataLoader
from train.batch_trainer.enkf_classification import BatchTrainer as EnKFTrainer
from train.batch_trainer.enkf_adaptive_train import BatchTrainer as EnkFAdaptiveTrainer
from adam_train.batch_trainer.classification import BatchTrainer as AdamTrainer  # Adjusted import for classification
from train.batch_trainer.enkf_iterative_train import ModelTrainer as DeepEKITrainer
import os
from datetime import datetime
from visualisation.plotter import Plotter

def benchmark_classification(trainer_class, model, dataset_loader, num_epochs=100, params=None):
    trainer = trainer_class(model=model, **params) if params else trainer_class(model=model)
    trainer.load_data(dataset_loader)

    trainer.train(num_epochs=num_epochs, is_plot_graph=0)
    trainer.evaluate_test()

    train_accuracy = trainer.train_accuracies
    val_accuracy = trainer.val_accuracies
    train_loss = trainer.train_losses
    val_loss = trainer.val_losses

    return train_accuracy, val_accuracy, train_loss, val_loss

if __name__ == '__main__':
    # dataset_loader = IrisDataLoader(num_features=2, set_standardize=False, batch_size=100000)
    # input_size = dataset_loader.get_feature_dim()
    # output_size = 3  # For Iris dataset (3 classes)

    file_path = './dataset/multi_class_classification_data.csv'  # Update with your file path
    dataset_loader = ClassificationDataLoader(file_path=file_path, set_standardize=True, test_size=0.2, val_size=0.1, batch_size=100000000)
    input_size, output_size = dataset_loader.get_feature_dim()

    # Define the models
    model_enkf = DNN(input_size=input_size, output_size=output_size)
    model_adam = DNN(input_size=input_size, output_size=output_size)
    model_deepeki = DNN(input_size=input_size, output_size=output_size)

    # Benchmark EnKFTrainer
    enkf_train_accuracy, enkf_val_accuracy, enkf_train_loss, enkf_val_loss = benchmark_classification(
        trainer_class=EnKFTrainer,
        model=model_enkf,
        dataset_loader=dataset_loader,
        num_epochs=100,
        params={"k": 100, "sigma": 0.01, "loss_type":"cross_entropy"},
    )

    # Benchmark BatchTrainer (using Adam optimizer)
    adam_train_accuracy, adam_val_accuracy, adam_train_loss, adam_val_loss = benchmark_classification(
        trainer_class=AdamTrainer,
        model=model_adam,
        dataset_loader=dataset_loader,
        num_epochs=100,
    )

    # Initialize Plotter
    plotter = Plotter()

    # Prepare data for comparison plotting
    accuracy_loss_data = [
        (enkf_train_accuracy, enkf_val_accuracy, enkf_train_loss, enkf_val_loss, 'EnKF'),
        (adam_train_accuracy, adam_val_accuracy, adam_train_loss, adam_val_loss, 'Adam'),
    ]

    # Plot comparison graph for accuracy
    plotter.plot_accuracy_loss_comparison(
        accuracy_loss_data=accuracy_loss_data,
    )
