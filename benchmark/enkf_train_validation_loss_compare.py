import torch
import matplotlib.pyplot as plt
from optimiser.enkf import EnKF
from model.dnn import DNN
from data.dataloader.regression_loader import OscillatoryDataLoader
from train.batch_trainer.enkf_train import BatchTrainer as EnKFTrainer
from visualisation.plotter import Plotter

def benchmark(trainer_class, model, dataset_loader, num_epochs=100, params=None):
    trainer = trainer_class(model=model, **params) if params else trainer_class(model=model)
    trainer.load_data(dataset_loader)
    trainer.train(num_epochs=num_epochs, is_plot_graph=0)
    trainer.evaluate_test()
    return trainer.train_loss, trainer.val_loss

if __name__ == '__main__':
    dataset_loader = OscillatoryDataLoader(batch_size=10000, val_size=0.2)
    input_size = dataset_loader.train_dataset.X.shape[1]
    output_size = dataset_loader.train_dataset.y.shape[1]

    enkf_params_list = [
        {"k": 20},
        {"k": 50},
        {"k": 100},
        {"k": 150},
        {"k": 200}
        # {"sigma":0.1},
        # {"sigma":0.01},
        # {"sigma":0.001},
        # {"sigma":0.0001},
        # {"gamma": 1e-1},
        # {"gamma": 1e-2},
        # {"gamma": 1e-3},
    ]

    plotter = Plotter()

    loss_data = []

    for params in enkf_params_list:
        model_enkf = DNN(input_size=input_size, output_size=output_size)
        enkf_train_loss, enkf_val_loss = benchmark(
            trainer_class=EnKFTrainer,
            model=model_enkf,
            dataset_loader=dataset_loader,
            num_epochs=1000,
            params=params,
        )

        # Add both training and validation losses to the same plot
        # loss_data.append((enkf_train_loss, f'Training Loss (K {params["k"]})'))
        # loss_data.append((enkf_val_loss, f'Validation Loss (K {params["k"]})'))
        loss_data.append((enkf_train_loss, enkf_val_loss, f'K {params["k"]}'))

    # Plot all comparisons in one graph
    # plotter.plot_benchmark_graph(
    #     loss_data=loss_data,
    #     log_scale=True,
    #     xlabel='Epochs',
    #     ylabel='Loss',
    #     # title='Training vs Validation Loss for Different K Values'
    # )

    # plotter.plot_train_validation_benchmark(
    #     loss_data=loss_data,
    #     log_scale=True,
    #     xlabel='Epochs',
    #     ylabel='Loss',
    #     # title='Training vs Validation Loss for Different K Values'
    # )

    plotter.plot_loss_difference_benchmark(
        loss_data=loss_data,
        #log_scale=True,
        xlabel='Epochs',
        #ylabel='Loss',
        # title='Training vs Validation Loss for Different K Values'
    )

    

    

