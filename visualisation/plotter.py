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


    def plot_accuracy_loss_comparison(self, accuracy_loss_data, xlabel='Epochs', ylabel_acc='Accuracy (%)', ylabel_loss='Loss', title_acc='Accuracy Comparison', title_loss='Loss Comparison', save_path_acc=None, save_path_loss=None):
        # Plot Accuracy Comparison
        plt.figure(figsize=self.figsize)
        
        color_cycler = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        for train_acc, val_acc, _, _, label in accuracy_loss_data:
            color = next(color_cycler)
            
            plt.plot(train_acc, linestyle='--', color=color, label=f'{label} Train Accuracy', linewidth=self.linewidth)
            plt.plot(val_acc, linestyle='-', color=color, label=f'{label} Validation Accuracy', linewidth=self.linewidth)
        
        plt.xlabel(xlabel, fontsize=self.fontsize)
        plt.ylabel(ylabel_acc, fontsize=self.fontsize)
        plt.title(title_acc, fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize - 2, loc='upper right')
        plt.grid(True)
        plt.tight_layout()

        if save_path_acc:
            plt.savefig(save_path_acc, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

        # Plot Loss Comparison
        plt.figure(figsize=self.figsize)
        
        color_cycler = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        for _, _, train_loss, val_loss, label in accuracy_loss_data:
            color = next(color_cycler)
            
            plt.plot(train_loss, linestyle='-', color=color, label=f'{label} Train Loss', linewidth=self.linewidth)
            plt.plot(val_loss, linestyle=':', color=color, label=f'{label} Validation Loss', linewidth=self.linewidth)
        
        plt.xlabel(xlabel, fontsize=self.fontsize)
        plt.ylabel(ylabel_loss, fontsize=self.fontsize)
        plt.title(title_loss, fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize - 2, loc='upper right')
        plt.grid(True)
        plt.tight_layout()

        if save_path_loss:
            plt.savefig(save_path_loss, dpi=300, bbox_inches='tight')
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