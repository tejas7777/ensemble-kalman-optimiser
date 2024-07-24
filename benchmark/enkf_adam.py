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
import os
from datetime import datetime

def benchmark(trainer_class, model, dataset_loader, num_epochs=100, params = None):
    trainer = trainer_class(model=model, **params) if params else trainer_class(model=model)
    trainer.load_data(dataset_loader)
    trainer.train(num_epochs=num_epochs, is_plot_graph=0)
    trainer.evaluate_test()
    return trainer.train_loss, trainer.val_loss

if __name__ == '__main__':
    #dataset_loader = OscillatoryDataLoader(test_size=0.3, val_size=0.3, batch_size=30)
    dataset_loader = ClassificationDataLoader(test_size=0.3, val_size=0.3, batch_size=1000)
    #input_size = dataset_loader.train_dataset.X.shape[1]
    #output_size = dataset_loader.train_dataset.y.shape[1]

    # Determine the input and output size from the feature dimensions
    input_size = dataset_loader.train_dataset.X.shape[1] 
    output_size = len(torch.unique(dataset_loader.train_dataset.y))

    # Define the models
    model_enkf = DNN(input_size=input_size, output_size=output_size)
    model_enkf_adaptive = DNN(input_size=input_size, output_size=output_size)
    model_adam = DNN(input_size=input_size, output_size=output_size)

    # Benchmark EnKFTrainer
    enkf_train_loss, enkf_val_loss = benchmark(
        trainer_class=EnKFTrainer,
        model=model_enkf,
        dataset_loader=dataset_loader,
        num_epochs=500,
        params={"k":100, "loss_type":"cross_entropy"},
    )

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
        num_epochs=500,
    )

    # Plotting the results
    plt.figure(figsize=(12, 6))
    #plt.plot(enkf_train_loss, label='EnKF Train Loss')
    plt.plot(enkf_val_loss, label='EnKF Validation Loss')
    #plt.plot(enkf_train_loss, label='EnKF Train Loss')
    plt.plot(enkf_adaptive_val_loss, label='EnKF Apative Validation Loss')
    #plt.plot(adam_train_loss, label='Adam Train Loss')
    plt.plot(adam_val_loss, label='Adam Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training and Validation Loss Comparison')
    plt.show()