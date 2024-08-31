import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimiser.enkf import EnKF
from model.dnn import DNNClassification as DNN 
from data.dataloader.iris_dataloader import IrisDataLoader
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

    return train_loss, val_loss

if __name__ == '__main__':
    dataset_loader = IrisDataLoader(num_features=2, set_standardize=False, batch_size=100000)
    input_size = dataset_loader.get_feature_dim()
    output_size = 3  # For Iris dataset (3 classes)

    # Define the models
    model_enkf = DNN(input_size=input_size, output_size=output_size)

    enkf_params_list = [
        # {"k": 20},
        # {"k": 50},
        # {"k": 100},
        # {"k": 150},
        # {"k": 200},
        # {"sigma": 0.01 },
        # {"sigma": 0.001 },
        # {"sigma": 0.0001 },
        # {"sigma": 0.00001 },
        {"gamma":1e-1},
        {"gamma":0.5},

    ]

    loss_data = []

    for i, params in enumerate(enkf_params_list):
        model_enkf = DNN(input_size=input_size, output_size=output_size)
        enkf_train_loss, enkf_val_loss = benchmark_classification(
            trainer_class=EnKFTrainer,
            model=model_enkf,
            dataset_loader=dataset_loader,
            num_epochs=200,
            params=params,
        )
        loss_data.append((enkf_val_loss, f'Gamma {params["gamma"]}'))

    # model_adam = DNN(input_size=input_size, output_size=output_size)
    # adam_train_loss, adam_val_loss = benchmark(
    #     trainer_class=AdamTrainer,
    #     model=model_adam,
    #     dataset_loader=dataset_loader,
    #     num_epochs=1000,
    # )
    # loss_data.append((adam_val_loss, 'Adam Validation Loss'))

    plotter = Plotter()
    plotter.plot_benchmark_graph(
        loss_data=loss_data,
        log_scale=True,
        xlabel='Epochs',
        ylabel='Loss',
        #title='Validation Loss Comparison'
    )