# This file is used as an interface to predict the ICB efficacy
# by getting patient's 6 indicators as input.
from likelihood import ode_likelihood
from train import MedicalDataset
import torch
from torch.utils.data import DataLoader
from utils import marginal_prob_std_fn,device,diffusion_coeff_fn
from model import ScoreNet
import numpy as np 
import pdb

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the pretrained model
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn,input_dim=7))
score_model = score_model.to(device)
ckpt = torch.load('ckpt.pth', map_location=device)
score_model.load_state_dict(ckpt) 
# Load train_files to get mean and std
file_path_train = 'train_type123.txt'
dataset_train = MedicalDataset(file_path_train)

# Get the patient's 6 indicators
def get_indicators():
    print("请输入患者的六维指标：")

    tmb = float(input("TMB (Tumor Mutation Burden): "))
    psth = float(input("PSTH (e.g., PD-L1 Score or other continuous marker): "))
    blood_albumin = float(input("Blood Albumin (g/dL): "))
    nlr = float(input("NLR (Neutrophil-to-Lymphocyte Ratio): "))
    age = float(input("Age: "))
    cancer_type = int(input("Cancer Type (encoded as integer, e.g., 0,1,2,...): "))

    indicators = [tmb, psth, blood_albumin, nlr, age, cancer_type, 0]
    indicators = torch.tensor([indicators], dtype=torch.float32)
    indicators = (indicators - dataset_train.mean_vals)/dataset_train.std_vals

    # 转换为 PyTorch 张量，形状为 (1, 6)
    return indicators

def get_indicators_from_web(features):
    indicators = features + [0]
    indicators = torch.tensor([indicators], dtype=torch.float32)
    indicators = (indicators - dataset_train.mean_vals)/dataset_train.std_vals
    
    return indicators

def compute_P(indicators, score_model=score_model, marginal_prob_std=marginal_prob_std_fn, diffusion_coeff=diffusion_coeff_fn,start_t=0.59):
    predicted_result=[]
    indicators = indicators.to(device)
    indicators_0 = indicators.clone()
    indicators_1 = indicators.clone()
    indicators_1[..., -1] = (1-dataset_train.mean_vals[-1])/dataset_train.std_vals[-1]

    # Compute likelihoods (probabilities) for both cases
    prob_0,_,_ = ode_likelihood(indicators_0, score_model, marginal_prob_std, diffusion_coeff,start_t=start_t, device=device)
    prob_1,_,_ = ode_likelihood(indicators_1, score_model, marginal_prob_std, diffusion_coeff,start_t=start_t, device=device)

    for i in range(len(indicators)):
        predicted_result.append(prob_1[i].item() / (prob_0[i].item()+prob_1[i].item()))
    return predicted_result[0]

if __name__=='__main__':
    indicators = get_indicators()
    pdb.set_trace()
    predicted_result = compute_P(indicators)
    print('Predicted result:', predicted_result)