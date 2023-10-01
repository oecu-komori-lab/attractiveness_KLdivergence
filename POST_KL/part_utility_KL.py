import numpy as np
import torch
import pandas as pd
import os

def KLdivergence(Mu1, S1, Mu2, S2): # Calculate KL divergence
    
    Mu1 = np.around(Mu1, decimals=6, out=None)
    Mu2 = np.around(Mu2, decimals=6, out=None)
    S1 = np.around(S1, decimals=6, out=None)
    S2 = np.around(S2, decimals=6, out=None)
    
    D = Mu1.shape[0] # Number of predicted means
    S2inv = np.linalg.inv(S2) # Compute the inverse of S2
    tmp = S2inv @ S1 # Matrix product of the inverse of S2 and S1
    Trace = np.trace(tmp) # Sum of the diagonal elements
    _ , logdet = np.linalg.slogdet(tmp)
    error = (Mu1 - Mu2).T @ S2inv @ (Mu1 - Mu2) # Transpose of (mean1 - mean2) @ inverse of S2 @ (mean1 - mean2)
    KL = 0.5 * (Trace - logdet + error - D)
    return KL

def part_all_KL(sessions, name_list):
    save_dir = "./result/part_KL/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    KL_all_list = np.empty([len(name_list)+1,2])
    
    for id, name in enumerate(name_list):
        
        beauty_mu = torch.load("./result/" + sessions[0] + "/" + name + "/" + name + "_all_PC_exKL_mu.pt").detach().numpy()
        cute_mu = torch.load("./result/" + sessions[1] + "/" + name + "/" + name + "_all_PC_exKL_mu.pt").detach().numpy()
        beauty_cov = torch.load("./result/" + sessions[0] + "/" + name + "/" + name + "_all_PC_exKL_covariance.pt").detach().numpy()
        cute_cov = torch.load("./result/" + sessions[1] + "/" + name + "/" + name + "_all_PC_exKL_covariance.pt").detach().numpy()
        
        cute_to_beauty_KL = KLdivergence(beauty_mu, beauty_cov, cute_mu, cute_cov)
        beauty_to_cute_KL = KLdivergence(cute_mu, cute_cov, beauty_mu, beauty_cov)
        
        KL_all_list[id,0] = cute_to_beauty_KL
        KL_all_list[id,1] = beauty_to_cute_KL
        
    beauty_mu = torch.load("./result/" + sessions[0] + "/all_PC_exKL_mu_mean.pt").detach().numpy()
    cute_mu = torch.load("./result/" + sessions[1] + "/all_PC_exKL_mu_mean.pt").detach().numpy()
    beauty_cov = torch.load("./result/" + sessions[0] + "/all_PC_exKL_covariance_mean.pt").detach().numpy()
    cute_cov = torch.load("./result/" + sessions[1] + "/all_PC_exKL_covariance_mean.pt").detach().numpy()
    
    cute_to_beauty_KL = KLdivergence(beauty_mu, beauty_cov, cute_mu, cute_cov)
    beauty_to_cute_KL = KLdivergence(cute_mu, cute_cov, beauty_mu, beauty_cov)
    
    KL_all_list[id+1,0] = cute_to_beauty_KL
    KL_all_list[id+1,1] = beauty_to_cute_KL

    index_name = name_list.tolist() + ["all"]
    df = pd.DataFrame(KL_all_list, index=index_name, columns=['CUTENESS to BEAUTY','BEAUTY to CUTENESS'])
    df.to_csv(save_dir + "part_KL_all_mean_KL.csv", float_format="%.6f")
    

if __name__=="__main__":
    SESSIONS = ["beauty","cute"]
    name_list = np.loadtxt("../data/name_list.csv", dtype="unicode") 
        
    part_all_KL(SESSIONS,name_list)