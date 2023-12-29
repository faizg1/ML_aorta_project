# We will merge 2 (or more) snapshot matrices to create a training set

import tensorflow as tf
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

from my_functions import *

#%% load data

# load flow field 1: -25% 
data_1 = np.load('D:/OneDrive - University College London/PhD/Catriona CFD data/Data/Python codes/snapshot_matrix_CFD_minus25_FL.npz')
X_1, X_1_mean, Xp_1, Xp_1_mean = ( data_1['X'], data_1['X_mean'], data_1['Xp'], data_1['Xp_mean'] )
X_1 = X_1 + X_1_mean
Xp_1 = Xp_1 + Xp_1_mean
del data_1, X_1_mean, Xp_1_mean

# load flow field 2: +0% (3DIVP)
data_2 = np.load('D:/OneDrive - University College London/PhD/Catriona CFD data/Data/Python codes/snapshot_matrix_CFD_plus25_FL.npz')
X_2, X_2_mean, Xp_2, Xp_2_mean = ( data_2['X'], data_2['X_mean'], data_2['Xp'], data_2['Xp_mean'] )
X_2 = X_2 + X_2_mean
Xp_2 = Xp_2 + Xp_2_mean

x, y, z, dt = ( data_2['x'], data_2['y'], data_2['z'], data_2['dt'] )
del data_2, X_2_mean, Xp_2_mean

#%% merge the snapshot matrix

X = np.hstack((X_1, X_2))
Xp = np.hstack((Xp_1, Xp_2))

del X_1, X_2

X_mean = np.mean(X, axis=1, keepdims=True)
Xp_mean = np.mean(Xp, axis=1, keepdims=True)

X = X - X_mean
Xp = Xp - Xp_mean


#%% save the merged snapshot matrix
# np.savez('snapshot_matrix_traning_minus25_3DIVP_FL.npz', X=X, X_mean=X_mean, Xp=Xp, Xp_mean=Xp_mean, x=x, y=y, z=z, dt=dt)

#%% perform POD

# for velocity field
U, sigma_vector, V = np.linalg.svd(X, full_matrices=False)
V = V.T
Sigma = np.diag(sigma_vector)
coeff = Sigma @ V.T

KE_fraction, KE_cumulative_fraction = POD_KE_plot(sigma_vector, 100)

# for pressure field
U_p, sigma_vector_p, V_p = np.linalg.svd(Xp, full_matrices=False)
V_p = V_p.T
Sigma_p = np.diag(sigma_vector_p)
coeff_p = Sigma_p @ V_p.T

#%% reconstruct ROM

error_list = []
error_p_list = []

# Loop over r = 1 to 30
for r in np.arange(1, 30+1, 1):
    
    # Reduced order approximation for velocity and pressure fields
    X_r = U[:, :r] @ Sigma[:r, :r] @ V[:, :r].T
    Xp_r = U_p[:, :r] @ Sigma_p[:r, :r] @ V_p[:, :r].T
    
    # Compute error for velocity and pressure
    error = np.sum(np.abs(X - X_r)) / np.sum(np.abs(X+X_mean)) * 100
    error_p = np.sum(np.abs(Xp - Xp_r)) / np.sum(np.abs(Xp+Xp_mean)) * 100
    
    # print("Velocity field error = ", round(error, 3), '%')
    # print("Pressure field error = ", round(error_p, 3), '%')
    
    # Append errors to their respective lists
    error_list.append(error)
    error_p_list.append(error_p)
    
    
#%% plot error curve
# Set font specifications
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20

plt.figure(figsize=(8,6))  # Increased figure size for better visualization

# Plotting the errors
plt.plot(np.arange(1, 30+1, 1), error_list, '-bo', markersize=6, markerfacecolor='b', markeredgecolor='b', label='Velocity Error')
plt.plot(np.arange(1, 30+1, 1), error_p_list, '-rs', markersize=6, markerfacecolor='r', markeredgecolor='r', label='Pressure Error')
plt.grid(True)
plt.xlabel("Mode number")
plt.ylabel("Reconstruction error (%)")
plt.legend(loc='best')
plt.xticks(np.arange(0, 30+1, 5)) # To show integer ticks on the x-axis


# Format y-axis tick labels to always show two digits
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))



#%% save
# np.savez('POD_traning_minus25_3DIVP_FL.npz', U=U, Sigma=Sigma, sigma_vector=sigma_vector, V=V, KE_fraction=KE_fraction, 
#           KE_cumulative_fraction=KE_cumulative_fraction, coeff=coeff, U_p=U_p, V_p=V_p, Sigma_p=Sigma_p, 
#           sigma_vector_p=sigma_vector_p, coeff_p=coeff_p, x=x, y=y, z=z, dt=dt)




