import matplotlib.pyplot as plt
import numpy as np
from src.logger import get_logger

logger = get_logger("Visualization")

def plot_energy_histogram(y_true, y_pred, file_name="energy_histogram.pdf"):
    """
    Plots a normalized histogram comparing DFT truth vs NN predictions.
    """
    colors = ["#d62728", "#1f77b4"]  # Red for 1st level, Blue for 2nd
    num_bins = 100
    range_xax = (-30, 0)

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
