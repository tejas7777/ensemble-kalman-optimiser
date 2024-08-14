import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import numpy as np

class Plotter:
    def __init__(self):
        self.figsize = (10, 6)
        self.fontsize = 12
        self.linewidth = 1.5
        self.setup_matlab_style()

    def setup_matlab_style(self):
        plt.style.use('default')
        sns.set_style("whitegrid", {'grid.linestyle': '--'})
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'axes.edgecolor': '#333333',
            'axes.labelcolor': '#333333',
            'text.color': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'grid.color': '#CCCCCC',
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
        })

    def plot_train_graph(self, train_losses: tuple[list[float], str], val_losses: tuple[list[float], str], 
                         log_scale=True, xlabel=None, ylabel=None, title=None, legend=True, save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(train_losses[0], label=train_losses[1], linewidth=self.linewidth)
        ax.plot(val_losses[0], label=val_losses[1], linewidth=self.linewidth)

        if log_scale:
            ax.set_yscale('log')

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.fontsize)
        if title:
            ax.set_title(title, fontsize=self.fontsize, fontweight='bold')

        if legend:
            ax.legend(fontsize=self.fontsize - 2, frameon=True, edgecolor='none')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_benchmark_graph(self, loss_data: list[tuple[list[float], str]], log_scale=True, xlabel='Epochs', ylabel='Loss', title=None, save_path=None):
        plt.figure(figsize=self.figsize)
        
        for losses, label in loss_data:
            plt.plot(losses, label=label, linewidth=self.linewidth)
        
        if log_scale:
            plt.yscale('log')

        plt.xlabel(xlabel, fontsize=self.fontsize)
        plt.ylabel(ylabel, fontsize=self.fontsize)
        if title:
            plt.title(title, fontsize=self.fontsize, fontweight='bold')

        plt.legend(fontsize=self.fontsize - 2, frameon=True, edgecolor='none')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


    def plot_train_validation_benchmark(self, loss_data, log_scale=True, xlabel='Epochs', ylabel='Loss', title=None, save_path=None):
        plt.figure(figsize=self.figsize)

        # Create a color cycler
        color_cycler = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        for train_loss, val_loss, label in loss_data:
            color = next(color_cycler)
            plt.plot(train_loss, linestyle='--', color=color, linewidth=self.linewidth, label=f'Training Loss ({label})')
            plt.plot(val_loss, linestyle='-', color=color, linewidth=self.linewidth, label=f'Validation Loss ({label})')

        if log_scale:
            plt.yscale('log')

        plt.xlabel(xlabel, fontsize=self.fontsize)
        plt.ylabel(ylabel, fontsize=self.fontsize)
        if title:
            plt.title(title, fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize - 2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_loss_difference_benchmark(self, loss_data, xlabel='Epochs', ylabel='Difference (Train - Validation)', log_scale = True, title=None, save_path=None):
        plt.figure(figsize=self.figsize)

        for train_loss, val_loss, label in loss_data:
            loss_diff = [train - val for train, val in zip(train_loss, val_loss)]
            plt.plot(loss_diff, label=f'Loss Difference ({label})', linewidth=self.linewidth)

        if log_scale:
            plt.yscale('log')

        plt.xlabel(xlabel, fontsize=self.fontsize)
        plt.ylabel(ylabel, fontsize=self.fontsize)
        if title:
            plt.title(title, fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize - 2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_loss_landscape(self, alphas, losses, xlabel='Alpha', ylabel='Loss', title='Loss Landscape via Linear Interpolation', save_path=None):
        plt.figure(figsize=self.figsize)
        plt.plot(alphas, losses, linewidth=self.linewidth)
        plt.xlabel(xlabel, fontsize=self.fontsize)
        plt.ylabel(ylabel, fontsize=self.fontsize)
        plt.title(title, fontsize=self.fontsize)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()