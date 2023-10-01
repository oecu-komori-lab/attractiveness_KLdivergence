# attractiveness_KLdivergence
## directory tree
```
.
├── POST_KL  :Evaluate KL Divergence using POST_KL utility function
│   ├── part_utility_KL.py  :Measure individual cuteness and beauty KL divergence
│   ├── result  :Utility functions of experiment participants
│   │   ├── beauty  :Beauty utility function
│   │   │   ├── all_PC_exKL_covariance_mean.pt
│   │   │   ├── all_PC_exKL_covariance_mean_list.pt
│   │   │   ├── all_PC_exKL_mu_mean.pt
│   │   │   ├── all_PC_exKL_mu_mean_list.pt
│   │   │   ├── all_PC_exKL_sigma2_mean.pt
│   │   │   ├── all_PC_exKL_sigma2_mean_list.pt
│   │   │   ├── participant1
│   │   │   │   ├── participant1_all_PC_exKL_covariance.pt
│   │   │   │   ├── participant1_all_PC_exKL_mu.pt
│   │   │   │   └── participant1_all_PC_exKL_sigma2.pt
│   │   │   |           ...
│   │   │   └── participant6
│   │   │       ├── participant6_all_PC_exKL_covariance.pt
│   │   │       ├── participant6_all_PC_exKL_mu.pt
│   │   │       └── participant6_all_PC_exKL_sigma2.pt
│   │   ├── cute  :Cuteness utility function
│   │   │   ├── all_PC_exKL_covariance_mean.pt
│   │   │   ├── all_PC_exKL_covariance_mean_list.pt
│   │   │   ├── all_PC_exKL_mu_mean.pt
│   │   │   ├── all_PC_exKL_mu_mean_list.pt
│   │   │   ├── all_PC_exKL_sigma2_mean.pt
│   │   │   ├── all_PC_exKL_sigma2_mean_list.pt
│   │   │   ├── participant1
│   │   │   │   ├── participant1_all_PC_exKL_covariance.pt
│   │   │   │   ├── participant1_all_PC_exKL_mu.pt
│   │   │   │   └── participant1_all_PC_exKL_sigma2.pt
│   │   │   |           ...
│   │   │   └── participant6
│   │   │       ├── participant6_all_PC_exKL_covariance.pt
│   │   │       ├── participant6_all_PC_exKL_mu.pt
│   │   │       └── participant6_all_PC_exKL_sigma2.pt
│   │   └── part_KL
│   │       └── part_KL_all_mean_KL.csv  :Results of cuteness and beauty KL divergence
│   └── utility_mean.py  :Code to estimate utility functions
|
├── POST_meshgrid  :Overall utility function estimation
│   ├── main.py  :Create models and perform estimation
│   ├── plot_mean_utility.py  :Generate utility function plots for each dimension
│   ├── postprocess_multi_dim.py  :Required functions
│   └── result
│       └── model  :Models for each experiment participant
│
├── README.md
└── data :two-alternative forced choice task data
    ├── beauty  :Beauty preference data
    ├── cute    :Cuteness preference data
    ├── main_pair_list.pt
    ├── main_result_data.pt  :Same exploration points
    └── name_list.csv  :List of experiment participants

```