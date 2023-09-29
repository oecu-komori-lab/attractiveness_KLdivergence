import numpy as np
import torch
import pandas as pd
import os
from KL_function import KLdivergence
from botorch.models import PairwiseGP

def pred_mu_cov(session, name_list): # KL用の効用の推定
    
    result = torch.load("./data/main_result_data.pt") # 参加者resultデータの読み込み
    
    for name in name_list:
        print("name:" ,name)
        result_dir = "./result/" + session + "/" + name + "/"
        
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir ,exist_ok=True)
            
        response = torch.load("./data/" + session + "/" + name + '/' + name + "_response.pt") 
        model = PairwiseGP(result, response)
        model.load_state_dict(torch.load("../POST_meshgrid/result/model/" + session + "/" + name + "/" + name + "_model_state.pth")) #modelの読み込み

        all_exKL_mu = model.posterior(result).mean.squeeze().unsqueeze(dim=1) # 予測平均
        all_exKL_sigma2 = model.posterior(result).variance.squeeze().unsqueeze(dim=1) #　予測共分散
        all_exKL_cov = model.posterior(result).covariance #　予測分散
        
        torch.save(all_exKL_mu, result_dir + name + "_all_PC_exKL_mu.pt")
        torch.save(all_exKL_sigma2, result_dir + name + "_all_PC_exKL_sigma2.pt")
        torch.save(all_exKL_cov, result_dir + name + "_all_PC_exKL_covariance.pt")
        
        
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
    for session in SESSIONS:
        pred_mu_cov(session,name_list)
        
    part_all_KL(SESSIONS,name_list)