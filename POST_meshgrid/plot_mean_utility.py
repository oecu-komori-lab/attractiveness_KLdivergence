import torch
import numpy as np
import matplotlib.pyplot as plt

def plot(point, dim, sessions):
    # Create subplots based on the dimensions
    fig, axes = plt.subplots(dim, 1, figsize=(24, 30))

    for ids in range(dim):
        index_list = list(range(8))
        mean_list = [i for i in index_list if i != ids]
        
        # Load mean and variance data for each session
        beauty_mean = torch.load("./pred/" + sessions[0] + "/all_mu_mean.pt").reshape(11, 11, 11, 11, 11, 11, 11, 11).detach().numpy()
        cute_mean = torch.load("./pred/" + sessions[1] + "/all_mu_mean.pt").reshape(11, 11, 11, 11, 11, 11, 11, 11).detach().numpy()
        beauty_var = torch.load("./pred/" + sessions[0] + "/all_sigma2_mean.pt").reshape(11, 11, 11, 11, 11, 11, 11, 11).detach().numpy()
        cute_var = torch.load("./pred/" + sessions[1] + "/all_sigma2_mean.pt").reshape(11, 11, 11, 11, 11, 11, 11, 11).detach().numpy()
        
        for m in mean_list[::-1]:
            cute_mean = cute_mean.mean(axis=m)
            beauty_mean = beauty_mean.mean(axis=m)
            beauty_var = beauty_var.mean(axis=m)
            cute_var = cute_var.mean(axis=m)
        
        cute_sd = np.sqrt(cute_var)
        beauty_sd = np.sqrt(beauty_var)
        
        # Plot the utility curves and uncertainty
        axes[ids].fill_between(point, cute_mean + cute_sd, cute_mean - cute_sd, color='blue', alpha=0.3)
        axes[ids].fill_between(point, beauty_mean + beauty_sd, beauty_mean - beauty_sd, color='orange', alpha=0.3)
        axes[ids].plot(point, beauty_mean.T, linewidth=2, color='orange', label=sessions[1])    
        axes[ids].plot(point, cute_mean.T, linewidth=2, color='blue', label=sessions[0])
        axes[ids].set_title("PC" + str(ids + 1))
        axes[ids].set_ylabel("UTILITY")
        axes[ids].set_xlim([-2, 2])
        axes[ids].set_ylim([-2, 2])
        
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=18, loc='upper right')  
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./img/mean_utility.pdf")

if __name__ == "__main__":
    # Generate points for plotting
    point = np.linspace(-2, 2, 11)
    SESSIONS = ["beauty", "cute"]
    DIM = 8
    # Call the plot function
    plot(point, DIM, SESSIONS)
