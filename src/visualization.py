import matplotlib.pyplot as plt
import numpy as np
from src.logger import get_logger
from scipy.stats import gaussian_kde

logger = get_logger("Visualization")

def energy_histogram_plot(y_true, y_pred, file_name="energy_histogram.pdf", num_bins = 100, range_xax = (-30, 0)):
    """
    Plots a normalized histogram comparing DFT truth vs NN predictions.
    """
    colors = ["#d62728", "#1f77b4"]  # Red for 1st level, Blue for 2nd

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate weights for normalization (as seen in post_processing.py)
    w_true1 = np.ones(len(y_true[:, 0])) / len(y_true[:, 0])
    w_true2 = np.ones(len(y_true[:, 1])) / len(y_true[:, 1])
    w_pred1 = np.ones(len(y_pred[:, 0])) / len(y_pred[:, 0])
    w_pred2 = np.ones(len(y_pred[:, 1])) / len(y_pred[:, 1])

    # Plot True Values (Steps)
    ax.hist(y_true[:, 0], bins=num_bins, range=range_xax, weights=w_true1, 
            label="1st, True", color=colors[0], histtype='step', linewidth=1.5)
    ax.hist(y_true[:, 1], bins=num_bins, range=range_xax, weights=w_true2, 
            label="2nd, True", color=colors[1], histtype='step', linewidth=1.5)
    
    # Plot Predictions (Filled)
    ax.hist(y_pred[:, 0], bins=num_bins, range=range_xax, weights=w_pred1, 
            label="1st, Pred", color=colors[0], alpha=0.3, histtype='stepfilled')
    ax.hist(y_pred[:, 1], bins=num_bins, range=range_xax, weights=w_pred2, 
            label="2nd, Pred", color=colors[1], alpha=0.3, histtype='stepfilled')

    ax.set_xlabel('Energy (mHartree)', fontsize=14)
    ax.set_ylabel('Normalized Frequency', fontsize=14)
    ax.set_ylim(0, 0.10)
    ax.legend(loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig(file_name)
    logger.info(f"Histogram saved as {file_name}")
    plt.close()
    
def energy_histogram(y_pred, file_name="energy_histogram.pdf", num_bins = 100, range_xax = (-30, 0)):
    """
    Plots a normalized histogram of NN predictions of EDA for MOF systems.
    """
    colors = ["#d62728", "#1f77b4"]  # Red for 1st level, Blue for 2nd

    fig, ax = plt.subplots(figsize=(8, 6))
    
    w_pred1 = np.ones(len(y_pred[:, 0])) / len(y_pred[:, 0])
    w_pred2 = np.ones(len(y_pred[:, 1])) / len(y_pred[:, 1])

    ax.hist(y_pred[:, 0], bins=num_bins, range=range_xax, weights=w_pred1, 
            label="1st, Pred", color=colors[0], alpha=0.3, histtype='stepfilled')
    ax.hist(y_pred[:, 1], bins=num_bins, range=range_xax, weights=w_pred2, 
            label="2nd, Pred", color=colors[1], alpha=0.3, histtype='stepfilled')

    ax.set_xlabel('Energy (mHartree)', fontsize=14)
    ax.set_ylabel('Normalized Frequency', fontsize=14)
    ax.set_ylim(0, 0.25)
    ax.legend(loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig(file_name)
    logger.info(f"Histogram saved as {file_name}")
    plt.close()

def correlation_plot(qq_true, qq_pred, file_name="correlation.png"):
    """Evaluates the model on the full dataset and generates a correlation plot."""
    #plt.rcParams['font.family'] = 'sans-serif'
    #rcParams['font.sans-serif'] = ['Tahoma']
    #plt.rcParams["font.family"] = "Times New Roman"
    
    print("Minimum predicted energy:", qq_pred.min())
    print("Maximum predicted energy:", qq_pred.max())
    
    qq_true = qq_true.flatten()
    qq_pred = qq_pred.flatten()

    # Density computation
    xy = np.vstack([qq_true, qq_pred])
    print("shape xy", xy.shape)
    z = gaussian_kde(xy)(xy)  # density values

    # Sort by density for better plotting
    idx = z.argsort()
    qq_true, qq_pred, z = qq_true[idx], qq_pred[idx], z[idx]
    
    # Plot correlation
    plt.figure(figsize=(10, 10))
    sc = plt.scatter(qq_true, qq_pred, c=z, cmap='plasma', s=22, edgecolor='none', alpha=0.5, label="Predictions")
    plt.plot(qq_true, qq_true, linestyle="-", color="#d62728", label="y=x")

    plt.xlim(-30, 0.5)
    plt.ylim(-30, 0.5)

    plt.xlabel('DFT energy (mHartree)', fontsize=22)
    plt.ylabel('NN energy (mHartree)', fontsize=22)

    cbar = plt.colorbar(sc, label='Density')
    # Change the font size of the label
    cbar.set_label('Density', fontsize=22)

    # Change the font size of the tick labels (the numbers)
    cbar.ax.tick_params(labelsize=18)

    plt.rcParams.update({'font.size': 22})
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(loc='upper left', frameon=False, fontsize=22)

    plt.tight_layout()
    #plt.savefig("correlation.pdf", dpi=50)
    plt.savefig(file_name)
    logger.info(f"Correlation saved as {file_name}")
    plt.close()

def loss_plot(train_losses, valid_losses, file_name="tv_loss.pdf"):
    """Plots losses vs epochs during training process"""
    epochs = range(len(train_losses))
    plt.figure(figsize=(8, 6))
    plt.yscale("log")
    plt.plot(epochs, train_losses, label='Train', color='C0', linewidth=2)
    plt.plot(epochs, valid_losses, label='Validation', color='C1', linewidth=2)
    
    # Plot the train loss with a line and a circle marker at every 10th epoch
    plt.scatter(epochs[::50], train_losses[::50], color='C0',
                     marker='o', edgecolor=None, label='_nolegend_')
    
    # Plot the validation loss with a line and a square marker at every 10th epoch
    plt.scatter(epochs[::50], valid_losses[::50], color='C1',
                     marker='s', edgecolor=None, label='_nolegend_')
    
    # Add labels, title, and legend
    plt.xlabel('Epochs', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    #plt.ylim(1e-2, 5e-1)
    #plt.xlim(-10, 360)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(loc='upper right', frameon=False, fontsize=18)
    #plt.grid(True)
    
    # Adjust layout and display plot
    plt.tight_layout()
    # Save the plot to a file for your manuscript
    # You can change the file name and format (e.g., .svg, .pdf)
    plt.savefig('tv_loss.pdf')
