import torch
import os
import numpy as np

def mean_pred(name_list, session):
    result_dir = "./result/" + session + "/"
    for num, name in enumerate(name_list):
        if num==0:
            all_exKL_mu = torch.load(result_dir + name + "/" + name + "_all_PC_exKL_mu.pt")
            all_exKL_sigma2 = torch.load(result_dir + name + "/" + name +"_all_PC_exKL_sigma2.pt")

        else:
            another_all_exKL_mu = torch.load(result_dir + name + "/" + name + "_all_PC_exKL_mu.pt")
            another_all_exKL_sigma2 = torch.load(result_dir + name + "/" + name +"_all_PC_exKL_sigma2.pt")
            
            all_exKL_mu = torch.cat([all_exKL_mu, another_all_exKL_mu], dim=1)
            all_exKL_sigma2 = torch.cat([all_exKL_sigma2, another_all_exKL_sigma2], dim=1)
            
    torch.save(all_exKL_mu, result_dir + "all_PC_exKL_mu_mean_list.pt")
    torch.save((all_exKL_mu.mean(axis=1)).reshape(-1,1), result_dir + "all_PC_exKL_mu_mean.pt")
    
    all_exKL_sigma2_mean = (torch.square(all_exKL_mu.to(torch.float32)).mean(axis=1) + 
                                all_exKL_sigma2.to(torch.float32).mean(axis=1)
                                - torch.square(all_exKL_mu.to(torch.float32).mean(axis=1))).reshape(-1,1)

    torch.save(all_exKL_sigma2, result_dir + "all_PC_exKL_sigma2_mean_list.pt")
    torch.save(all_exKL_sigma2_mean, result_dir + "all_PC_exKL_sigma2_mean.pt")

def mean_pred_cov(session, name_list):
    result_dir = "./result/" + session + "/"
    # 全次元　KL用のcov
    mean_mu = torch.load(result_dir + "all_PC_exKL_mu_mean.pt")
    mean_mu = mean_mu @ mean_mu.T
    
    for num, name in enumerate(name_list):
        if num == 0:
            exKL_cov = torch.load(result_dir + name + "/" + name + "_all_PC_exKL_covariance.pt")
            exKL_mu = torch.load(result_dir + name + "/" + name + "_all_PC_exKL_mu.pt")
            exKL_mu = exKL_mu @ exKL_mu.T
            exKL_cov_mean = (exKL_cov + exKL_mu).unsqueeze(dim=0)
            
        else:
            another_exKL_cov = torch.load(result_dir + name + "/" + name + "_all_PC_exKL_covariance.pt")
            another_exKL_mu = torch.load(result_dir + name + "/" + name + "_all_PC_exKL_mu.pt")
            another_exKL_mu = another_exKL_mu @ another_exKL_mu.T
            
            another_cov_mean = (another_exKL_cov + another_exKL_mu).unsqueeze(dim=0)
            
            exKL_cov_mean = torch.cat([exKL_cov_mean, another_cov_mean], dim=0)
            
    cov_mean = exKL_cov_mean.mean(axis=0).squeeze(dim=0) - mean_mu
    
    torch.save(exKL_cov_mean, result_dir + "all_PC_exKL_covariance_mean_list.pt")
    torch.save(cov_mean, result_dir + "all_PC_exKL_covariance_mean.pt")
    
    
if __name__=="__main__":
    name_list = np.loadtxt("../data/name_list.csv", dtype="unicode")
    SESSIONS = ["beauty","cute"]
        
    for session in SESSIONS:
        mean_pred(name_list, session)
        mean_pred_cov(session, name_list)