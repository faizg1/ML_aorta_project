# my_functions.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
 #  ___  ___   ___    _
 # | _ \| _ \ / __|  /_\  
 # |   /|  _/| (__  / _ \ 
 # |_|_\|_|   \___|/_/ \_\                     
###############################################################################

### This code is taken from this blog: https://freshprinceofstandarderror.com/ai/robust-principal-component-analysis/
### It was modified a little to take lamnbda as another input

# RPCA separate the data matrix X into 
# 1.) a low-rank matrix L that represents X, and 
# 2.) a sparse matrix S that contains outliners

# lambda can be used as a tuning parameter
# low lambda = high filtering
# high lambda = low fitering

# Example 

# lambd = 1/np.sqrt(np.max(np.shape(X))); # the recommended value of lambda 
# [L,S] = RPCA(X, lambd);


def shrink(X,tau):
    Y = np.abs(X)-tau
    return np.sign(X) * np.maximum(Y,np.zeros_like(Y))

def SVT(X,tau):
    U,S,VT = np.linalg.svd(X,full_matrices=0)
    out = U @ np.diag(shrink(S,tau)) @ VT
    return out

def RPCA(X, lambd):
    n1,n2 = X.shape
    mu = n1*n2/(4*np.sum(np.abs(X.reshape(-1))))
    thresh = 10**(-7) * np.linalg.norm(X, ord = 'fro')
    
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    L = np.zeros_like(X)
    count = 0
    while (np.linalg.norm(X-L-S) > thresh) and (count < 1000):
        L = SVT(X-S+(1/mu)*Y,1/mu)
        S = shrink(X-L+(1/mu)*Y,lambd/mu)
        Y = Y + mu*(X-L-S)
        count += 1
    return L,S

###############################################################################
 #  ___   ___   ___    _  __ ___         _       _   
 # | _ \ / _ \ |   \  | |/ /| __|  _ __ | | ___ | |_ 
 # |  _/| (_) || |) | | ' < | _|  | '_ \| |/ _ \|  _|
 # |_|   \___/ |___/  |_|\_\|___| | .__/|_|\___/ \__|
 #                                |_|                
###############################################################################


def POD_KE_plot(sigma_vector, num_mode):

    # Energy in POD coherent structures
    # computes KE fraction in each POD mode
    mode = np.arange(1, num_mode+1)
    KE_fraction = np.zeros(num_mode)
    KE_cumulative_fraction = np.zeros(num_mode)
    
    for i in mode:
        KE_fraction[i-1] = sigma_vector[i-1]**2/np.sum(sigma_vector**2)
        KE_cumulative_fraction[i-1] = np.sum(sigma_vector[0:i]**2)/np.sum(sigma_vector**2)

    plt.figure()
    plt.plot(mode, KE_fraction*100, '-ko', markersize=5, markerfacecolor='k', markeredgecolor='k')
    # plt.ylim([0, 100.01])
    plt.ylabel('Fluctuation kinetic energy fraction (%)')
    plt.xlabel('Number of mode')
    plt.grid(True)
    
    plt.figure()
    plt.plot(mode, KE_cumulative_fraction*100, '-ko', markersize=5, markerfacecolor='k', markeredgecolor='k')
    # plt.ylim([85, 100.01])
    plt.xlabel('Number of mode')
    plt.ylabel('Cumulative fluctuation kinetic energy fraction (%)')
    plt.grid(True)
    
    return KE_fraction, KE_cumulative_fraction


###############################################################################               _                             _                           
 #   __ _   ___ | |_     ___  _ __    ___   ___ | |_  _ __  _   _  _ __ ___  
 #  / _` | / _ \| __|   / __|| '_ \  / _ \ / __|| __|| '__|| | | || '_ ` _ \ 
 # | (_| ||  __/| |_    \__ \| |_) ||  __/| (__ | |_ | |   | |_| || | | | | |
 #  \__, | \___| \__|   |___/| .__/  \___| \___| \__||_|    \__,_||_| |_| |_|
 #  |___/                    |_|                                             
###############################################################################

def get_spectrum(x, sampling_rate):
    """
    Compute the power spectrum of a time-domain signal using the FFT.
    
    Args:
        x (ndarray): The time-domain signal to compute the spectrum of.
        sampling_rate (float): The sampling rate of the signal in Hz.
        
    Returns:
        power (ndarray): The power spectrum of the signal.
        f (ndarray): The corresponding frequency range of the power spectrum.
    """
    # Pad signal to make it periodic
    x = np.hstack([x, x[0]])
    
    # Compute FFT of signal
    y = np.fft.fft(x)
    n = len(x) # number of samples
    
    # Compute frequency range
    f = (sampling_rate / 2) * np.linspace(0, 1, n//2 + 1)
    
    # Compute power spectrum
    P2 = np.abs(y / n)
    P1 = P2[0:n//2 + 1]
    P1[1:-1] = 2 * P1[1:-1]

    power = P1
    
    return power, f
  
#####################################################################################
  

# Generate a dataset of sliding windows from the given 2D data.

# This function takes as input a 2D numpy array and a specified window size. It 
# slides a "window" of the given size over the data, and for each position of 
# the window, it adds the elements covered by the window to the output dataset. 
# The output is a 3D numpy array where each element is a sequence of 'window_size'
# elements from the input data.

# Inputs:
# data (np.array): The input 2D numpy array.
# window_size (int): The size of the sliding window.

# Returns:
# np.array: A 3D numpy array where each element is a window of elements from the input data.

def sliding_windows(data, window_size):
    # Initialize the list that will hold the sliding windows
    X = []
    
    # Slide a "window" of size window_size over the data
    for i in range(len(data) - window_size):
        # For each position, get the window_size elements following it
        # and append to the list X
        X.append(data[i:i + window_size])
        
    # Convert the list of windows back into a numpy array
    return np.array(X)




