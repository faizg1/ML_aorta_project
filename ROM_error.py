import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import matplotlib.lines as mlines

class CustomScalarFormatter(ScalarFormatter):
    def _set_format(self):  # Override the method
        self.format = "%1.1f"

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense

from scipy.io import loadmat
from scipy.stats import zscore

from keras.preprocessing.sequence import TimeseriesGenerator


from my_functions import *


#%% load data

# load the test flow field
temp = np.load('D:/OneDrive - University College London/PhD/Catriona CFD data/Data/Python codes/snapshot_matrix_CFD_plus25_FL.npz')
X, X_mean, Xp, Xp_mean = ( temp['X'], temp['X_mean'], temp['Xp'], temp['Xp_mean'] )
X = X + X_mean
Xp = Xp + Xp_mean
del temp, X_mean, Xp_mean

# load data needed from the traning set
# POD modes (basis vectors)
temp = np.load('POD_traning_minus25_plus25_FL.npz')
U, Up = ( temp['U'], temp['U_p'] )
del temp

# X_mean, Xp_mean
temp = np.load('snapshot_matrix_traning_minus25_plus25_FL.npz')
X_mean, Xp_mean = ( temp['X_mean'], temp['Xp_mean'] )
del temp

X = X - X_mean
Xp = Xp - Xp_mean

#%%

# Number of modes in ROMs
r = 22
rp = 2

# reconstruct the snapshot matrix (Project test dataset onto the ROMs' basis)
X_r = U[:, :r] @ ( U.T @ X )[:r, :]
Xp_r = Up[:, :rp] @ ( Up.T @ Xp )[:rp, :]


# Compute error for velocity and pressure 

# Error1
error = np.sum(np.abs(X - X_r)) / np.sum(np.abs(X + X_mean)) * 100
error_p = np.sum(np.abs(Xp - Xp_r)) / np.sum(np.abs(Xp + Xp_mean)) * 100

# # Error2
# error = np.mean(np.abs(X - X_r))/(( np.abs(np.min(X+X_mean)) + np.abs(np.max(X+X_mean)))/2) * 100
# error_p = np.mean(np.abs(Xp - Xp_r))/(( np.abs(np.min(Xp+Xp_mean)) + np.abs(np.max(Xp+Xp_mean)))/2) * 100

# # Error3
# error = np.mean(np.abs(X - X_r))/(0.2329) * 100
# error_p = np.mean(np.abs(Xp - Xp_r))/(13693.56) * 100

print("Velocity field error = ", round(error, 3), '%')
print("Pressure field error = ", round(error_p, 3), '%')



#%% Plot coeff in time

coeff = U.T @ X
coeff_p = Up.T @ Xp

nt = 127
t = np.arange(nt)
dt = 0.005

class CustomScalarFormatter(ScalarFormatter):
    def _set_format(self):  # Override the method
        self.format = "%1.1f"

for mode in np.arange(1, rp+1, 1):
    plt.rcParams["font.size"] = 24
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(8, 5.5))
    
    t = np.arange(nt)
    
    plt.plot(t*dt, coeff_p[mode-1, t], '-k', linewidth=3)
    plt.grid(True)
    
    plt.tick_params(labelsize=26)
    plt.ylabel(r'$b_{%s}(t)$' % mode, fontsize=32, math_fontfamily='cm')
    plt.xlabel('$t(s)$', fontsize=32, math_fontfamily='cm')
    
    # Set the y-axis to scientific notation with one decimal place
    formatter = CustomScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.grid(True)
    plt.tight_layout()  # Adjusts subplot params to give specified padding
    plt.show()
    
    
#%% This is to plot both coeffs

# 1 is +25%
coeff_1 = coeff
coeff_p_1 = coeff_p

# load the test flow field
temp = np.load('D:/OneDrive - University College London/PhD/Catriona CFD data/Data/Python codes/snapshot_matrix_CFD_minus25_FL.npz')
X, X_mean, Xp, Xp_mean = ( temp['X'], temp['X_mean'], temp['Xp'], temp['Xp_mean'] )
X_2 = X + X_mean
Xp_2 = Xp + Xp_mean
del temp, X_mean, Xp_mean

# X_mean, Xp_mean
temp = np.load('snapshot_matrix_traning_minus25_plus25_FL.npz')
X_mean, Xp_mean = ( temp['X_mean'], temp['Xp_mean'] )
del temp

coeff_2 = U.T @ (X_2-X_mean)
coeff_p_2 = Up.T @ (Xp_2-Xp_mean)


coeff_1 = coeff_1[:, :nt]
coeff_p_1 = coeff_p_1[:, :nt]

coeff_2 = coeff_2[:, :nt]
coeff_p_2 = coeff_p_2[:, :nt]

class CustomScalarFormatter(ScalarFormatter):
    def _set_format(self):  # Override the method
        self.format = "%1.1f"

for mode in np.arange(1, r+1, 1):
    plt.rcParams["font.size"] = 24
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(8, 5.5))
    
    t = np.arange(nt)
    
    plt.plot(t*dt, coeff_1[mode-1, t], linewidth=3, label='+25%')
    plt.plot(t*dt, coeff_2[mode-1, t], linewidth=3, label='-25%')
    plt.grid(True)
    plt.legend()
    
    plt.tick_params(labelsize=26)
    plt.ylabel(r'$a_{%s}(t)$' % mode, fontsize=32, math_fontfamily='cm')
    plt.xlabel('$t(s)$', fontsize=32, math_fontfamily='cm')
    
    # Set the y-axis to scientific notation with one decimal place
    formatter = CustomScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.grid(True)
    plt.tight_layout()  # Adjusts subplot params to give specified padding
    plt.show()




