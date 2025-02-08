import numpy as np
import matplotlib.pyplot as plt
from utils import *

%matplotlib inline

X_train, X_val, y_val = load_data()

def estimate_gaussian(X): 
    m, n = X.shape
     
    mu = 1/m * np.sum(X, axis=0)
    var = 1/m * np.sum((X - mu)**2, axis=0)
        
    return mu, var

def select_threshold(y_val, p_val): 
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = (p_val < epsilon)
        
        tp = np.sum((y_val == 1) & (predictions == 1))
        fp = np.sum((y_val == 0) & (predictions == 1))
        fn = np.sum((y_val == 1) & (predictions == 0))
        
        prec = tp / (tp + fp)
        rec = tp / (tp + fn) 
        F1 = 2*prec*rec/(prec + rec)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1

# Apply the same steps to the larger dataset
# load the dataset
X_train_high, X_val_high, y_val_high = load_data_multi()

# Estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)

# Evaluate the probabilites for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)

# Evaluate the probabilites for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

# Find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))