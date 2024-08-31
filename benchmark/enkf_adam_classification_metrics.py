import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from optimiser.enkf import EnKF
from model.dnn import DNNClassification as DNN  # Adjusted for classification
from data.dataloader.iris_dataloader import IrisDataLoader  # Example data loader for classification
from train.batch_trainer.enkf_classification import BatchTrainer as EnKFTrainer
from adam_train.batch_trainer.classification import BatchTrainer as AdamTrainer  # Adjusted import for classification
from visualisation.plotter import Plotter
from sklearn.metrics import confusion_matrix, classification_report

def benchmark_classification(trainer_class, model, dataset_loader, num_epochs=100, params=None):
    trainer = trainer_class(model=model, **params) if params else trainer_class(model=model)
    trainer.load_data(dataset_loader)

    trainer.train(num_epochs=num_epochs, is_plot_graph=0)
    cm, report, test_accuracy = trainer.evaluate_test()

    train_accuracy = trainer.train_accuracies
    val_accuracy = trainer.val_accuracies
    train_loss = trainer.train_losses
    val_loss = trainer.val_losses

    return train_accuracy, val_accuracy, train_loss, val_loss, cm, report

if __name__ == '__main__':
    dataset_loader = IrisDataLoader(num_features=2, set_standardize=False, batch_size=100000)
    input_size = dataset_loader.get_feature_dim()
    output_size = 3  # For Iris dataset (3 classes)

    # Define the models
    model_enkf = DNN(input_size=input_size, output_size=output_size)
    model_adam = DNN(input_size=input_size, output_size=output_size)

    # Benchmark EnKFTrainer
    enkf_train_accuracy, enkf_val_accuracy, enkf_train_loss, enkf_val_loss, enkf_cm, enkf_report = benchmark_classification(
        trainer_class=EnKFTrainer,
        model=model_enkf,
        dataset_loader=dataset_loader,
        num_epochs=100,
        params={"k": 50, "sigma": 0.01, "loss_type": "cross_entropy"},
    )

    # Benchmark BatchTrainer (using Adam optimizer)
    adam_train_accuracy, adam_val_accuracy, adam_train_loss, adam_val_loss, adam_cm, adam_report = benchmark_classification(
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

    # Plot comparison graph for accuracy and loss
    plotter.plot_accuracy_loss_comparison(
        accuracy_loss_data=accuracy_loss_data,
    )

    # Print Confusion Matrices and Classification Reports
    print("EnKF Confusion Matrix:\n", enkf_cm)
    print("EnKF Classification Report:\n", enkf_report)

    print("Adam Confusion Matrix:\n", adam_cm)
    print("Adam Classification Report:\n", adam_report)
