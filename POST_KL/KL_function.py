import numpy as np

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
