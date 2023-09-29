# attractiveness_KLdivergence


.
├── KL_function.py
├── README.md
├── data
│   ├── beauty
│   ├── cute
│   ├── main_pair_list.pt
│   ├── main_result_data.pt
│   └── name_list.csv  part_name 
├── main.py
├── make_data.py
├── part_utility_KL.py
├── postprocess_multi_dim.py
├── result
│   ├── beauty
│   │   ├── all_PC_exKL_covariance_mean.pt
│   │   ├── all_PC_exKL_covariance_mean_list.pt
│   │   ├── all_PC_exKL_mu_mean.pt
│   │   ├── all_PC_exKL_mu_mean_list.pt
│   │   ├── all_PC_exKL_sigma2_mean.pt
│   │   ├── all_PC_exKL_sigma2_mean_list.pt
│   │   ├── participant1
│   │   │   ├── participant1_all_PC_exKL_covariance.pt
│   │   │   ├── participant1_all_PC_exKL_mu.pt
│   │   │   └── participant1_all_PC_exKL_sigma2.pt
                ...
│   │   └── participant6
│   │       
│   ├── cute
│   │   ├── all_PC_exKL_covariance_mean.pt
│   │   ├── all_PC_exKL_covariance_mean_list.pt
│   │   ├── all_PC_exKL_mu_mean.pt
│   │   ├── all_PC_exKL_mu_mean_list.pt
│   │   ├── all_PC_exKL_sigma2_mean.pt
│   │   ├── all_PC_exKL_sigma2_mean_list.pt
│   │   ├── participant1
│   │   │   ├── participant1_all_PC_exKL_covariance.pt
│   │   │   ├── participant1_all_PC_exKL_mu.pt
│   │   │   └── participant1_all_PC_exKL_sigma2.pt
                ...
│   │   └── participant6
│   │       ├── participant6_all_PC_exKL_covariance.pt
│   │       ├── participant6_all_PC_exKL_mu.pt
│   │       └── participant6_all_PC_exKL_sigma2.pt
│   └── part_KL
│       └── part_KL_all_mean_KL.csv
└── utility_mean.py