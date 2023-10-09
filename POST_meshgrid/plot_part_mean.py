import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def plot(point, name_list, dim, sessions):
    cmap = plt.get_cmap("tab10")
    img_dir = "./img/"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
        
    # Create subplots based on the dimensions
    fig, axes = plt.subplots(dim, 2, figsize=(10, 16))
    for session_id , session in enumerate(sessions):
        
        for ids in range(dim):
            print(ids)
            index_list = list(range(8))
            for nid, name in enumerate(name_list):
                mean_list = [i for i in index_list if i != ids]
                # Load mean and variance data for each session
                part_mean = torch.load("./pred/" + session + "/" + name + "_mu.pt").reshape(11, 11, 11, 11, 11, 11, 11, 11).detach().numpy()
                
                for m in mean_list[::-1]:
                    part_mean = part_mean.mean(axis=m)
                    
                axes[ids,session_id].plot(point, part_mean.T, label=name, color=cmap(nid))    
                # axes[ids].plot(point, cute_mean.T, linewidth=2, color='blue', label=sessions[0])
                axes[ids,session_id].set_title("PC" + str(ids + 1))
                axes[ids,session_id].set_ylabel("UTILITY")
                axes[ids,session_id].set_xlim([-2, 2])
                axes[ids,session_id].set_ylim([-2, 2])

    # Add legend        
    handles, labels = axes[0,0].get_legend_handles_labels()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left' ,handles=handles[:6], labels=labels[:6])
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("BEAUTY                     CUTENESS", fontsize=18 )
    plt.savefig(img_dir + "/part_mean_utility.pdf")
        
if __name__ == "__main__":
    # Generate points for plotting
    point = np.linspace(-2, 2, 11)
    SESSIONS = ["beauty", "cute"]
    DIM = 8
    name_list = np.loadtxt("../data/name_list.csv", dtype="unicode")
    # Call the plot function
    plot(point, name_list, DIM, SESSIONS)