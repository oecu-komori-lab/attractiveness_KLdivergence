import numpy as np
import torch
import pandas as pd
import os
from KL_function import KLdivergence

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

    print(name_list)
    index_name = name_list.tolist() + ["all"]
    df = pd.DataFrame(KL_all_list, index=index_name, columns=['CUTENESS to BEAUTY','BEAUTY to CUTENESS'])
    df.to_csv(save_dir + "part_KL_all_mean_KL.csv", float_format="%.6f")
    

if __name__=="__main__":
    SESSIONS = ["beauty","cute"]
    name_list = np.loadtxt("../data/name_list.csv", dtype="unicode")
    part_all_KL(SESSIONS,name_list)