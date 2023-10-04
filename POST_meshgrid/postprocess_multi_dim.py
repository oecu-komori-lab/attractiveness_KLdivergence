import os
import shutil
import torch
from botorch.models import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.optim.fit import fit_gpytorch_torch

def point_mesh():
    # Generate a point mesh for utility evaluation
    start_ = -1.0
    end_ = 1.0
    step_ = 0.2
    end_ = end_ + step_
    point = torch.arange(start_, end_, step_)
    x1, x2, x3, x4, x5, x6, x7, x8 = torch.meshgrid(point, point, point, point, point, point, point, point)
    x_mesh = torch.stack((x1, x2, x3, x4, x5, x6, x7, x8), dim=-1)
    torch.save(x_mesh.reshape(-1, 8), "./point_mesh.pt")

def model_create(name_list, session):
    # Create and save GP models for each participant
    result_dir = "./result/" + str(session) + "/model/"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    result = torch.load("../data/" + str(session) + "/" + "main_result_data.pt")

    for i in range(len(name_list)):
        response = torch.load("../data/" + str(session) + "/" + name_list[i] + "/" + name_list[i] + "_response.pt")
        model = PairwiseGP(result, response)
        mll = PairwiseLaplaceMarginalLogLikelihood(model)
        fit_gpytorch_torch(mll, options={"maxiter": 100, "disp": True, "lr": 0.01})
        state_dict = model.state_dict()
        torch.save(state_dict, result_dir + name_list[i] + "_model_state.pth")

def predict_dist(name_list, session, length=1000, size=214359):
    # Predict distributions for each participant
    result_dir = "./pred/" + str(session) + "/"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    points_mesh = torch.load("./point_mesh.pt")
    result = torch.load("../data/main_result_data.pt")

    for i in range(len(name_list)):
        result_dir_each = result_dir + name_list[i] + "/dist/"

        if not os.path.isdir(result_dir_each):
            os.makedirs(result_dir_each)

        response = torch.load("../data/" + str(session) + "/" + name_list[i] + "/" + name_list[i] + "_response.pt")
        model = PairwiseGP(result, response)
        model.load_state_dict(torch.load("./result/model/" + str(session) + "/" + name_list[i] + "/" + name_list[i] + "_model_state.pth"))

        for j in range(0, len(points_mesh), length):
            points = points_mesh[j: j + length]
            mu = model.posterior(points).mean
            sigma2 = model.posterior(points).variance
            if len(mu) == length:
                torch.save(mu.to(torch.float32), result_dir_each + str(j // length) + "_mu.pt")
                torch.save(sigma2.to(torch.float32), result_dir_each + str(j // length) + "_sigma2.pt")
                mu = None
                sigma2 = None
            else:
                torch.save(mu.to(torch.float32), result_dir_each + "last_mu.pt")
                torch.save(sigma2.to(torch.float32), result_dir_each + "last_sigma2.pt")
                mu = None
                sigma2 = None
        concatenate_dist(name=name_list[i], session=session, size=size)

def concatenate_dist(name, session, size):
    # Concatenate predicted distributions for a participant
    result_dir = "./pred/" + str(session) + "/"
    result_dir_each = result_dir + name + "/dist/"
    mu_0 = torch.load(result_dir_each + "0_mu.pt").reshape(-1, 1)
    sigma2_0 = torch.load(result_dir_each + "0_sigma2.pt").reshape(-1, 1)
    for j in range(size):
        if j == 0:
            mu = mu_0
            sigma2 = sigma2_0
        elif j < (size - 1):
            mu_another = torch.load(result_dir_each + str(j) + "_mu.pt").reshape(-1, 1)
            mu = torch.vstack((mu, mu_another))
            sigma2_another = torch.load(result_dir_each + str(j) + "_sigma2.pt").reshape(-1, 1)
            sigma2 = torch.vstack((sigma2, sigma2_another))
        else:
            mu_another = torch.load(result_dir_each + "last_mu.pt").reshape(-1, 1)
            mu = torch.vstack((mu, mu_another))
            sigma2_another = torch.load(result_dir_each + "last_sigma2.pt").reshape(-1, 1)
            sigma2 = torch.vstack((sigma2, sigma2_another))
    remove_intermediate_dist(name=name, session=session)
    torch.save(mu, result_dir + name + "_mu.pt")
    torch.save(sigma2, result_dir + name + "_sigma2.pt")

def remove_intermediate_dist(name, session):
    # Remove intermediate predicted distributions
    remove_dir = "./pred/" + str(session) + "/" + name + "/dist/"
    if os.path.isdir(remove_dir) is True:
        shutil.rmtree(remove_dir)

def main_synthesize(name_list, session):
    # Synthesize main results based on predicted distributions
    result_dir = "./pred/" + str(session) + "/"
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    mu_list = torch.load(result_dir + name_list[0] + "_mu.pt")
    sigma2_list = torch.load(result_dir + name_list[0] + "_sigma2.pt")

    for i in range(1, len(name_list)):
        mu_list_another = torch.load(result_dir + name_list[i] + "_mu.pt")
        mu_list = torch.hstack((mu_list, mu_list_another))
        sigma2_list_another = torch.load(result_dir + name_list[i] + "_sigma2.pt")
        sigma2_list = torch.hstack((sigma2_list, sigma2_list_another))
    torch.save(mu_list.to(torch.float32), result_dir + "all_mu_list.pt")
    torch.save(mu_list.to(torch.float32).mean(axis=1), result_dir + "all_mu_mean.pt")
    torch.save(sigma2_list.to(torch.float32), result_dir + "all_sigma2_list.pt")
    all_sigma2_mean = (torch.square(mu_list.to(torch.float32)).mean(axis=1)
                       + sigma2_list.to(torch.float32).mean(axis=1)
                       - torch.square(mu_list.to(torch.float32).mean(axis=1))
                       )
    torch.save(all_sigma2_mean.to(torch.float32), result_dir + "all_sigma2_mean.pt")
