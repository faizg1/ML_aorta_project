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

# load Inlet flowrate of the test flow field
flowrate = np.load('../InletFLowRate_3DIVP.npz')['flowrate']

# load the test flow field
temp = np.load('../snapshot_matrix_CFD_3DIVP_FL.npz')
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

# calculate the coeff by projection
coeff = U.T @ (X - X_mean)
coeff_p = Up.T @ (Xp - Xp_mean)


# load mean and std for normalisation
temp = np.load('mean_and_std_for_Network0_and1.npz')
mean_feature, std_feature = ( temp['mean_feature'], temp['std_feature'] )
mean_label, std_label = ( temp['mean_label'], temp['std_label'] )
del temp

#%% define the datasets

window_size = 4

nt = len(coeff) # number of temporal datapoints

# feature for the first sequence
feature = pd.DataFrame(flowrate)
num_feature = len(feature.columns)


# feature and labels
num_modes_v = 22 # number of velocity POD modes included we want to predict (for the reconstruction)
num_modes_p = 2 # number of pressure POD modes included we want to predict (for the reconstruction)
num_label = num_modes_v + num_modes_p
num_feature = num_label

label_velocity = np.transpose(coeff[0: num_modes_v, 0: nt])
label_velocity = pd.DataFrame(label_velocity)

label_pressure = np.transpose(coeff_p[0: num_modes_p, 0: nt])
label_pressure = pd.DataFrame(label_pressure)

label = pd.concat([label_velocity, label_pressure], axis=1)
label.columns = range(label.columns.size) # reset column names to numbers


# # standardise the data
# mean_feature = feature.mean()
# mean_label = label.mean()

# std_feature = feature.std()
# std_label = label.std()

feature = (feature - mean_feature)/std_feature
# label = (label - mean_label)/std_label

#%% use trained Network0 to predict the first sequence

tic = time.time()

model_network0 = load_model('Network0.h5')

first_sequence_flowrate = (feature.to_numpy())[0: window_size].reshape(1, window_size, 1)

first_sequence_coeff = model_network0.predict(first_sequence_flowrate, verbose=0)[0]

#%% use trained Network1 to predict the next steps sequentialy

model_network1 = load_model('Network1.h5')

prediction = np.zeros((label.shape[0], label.shape[1]))

# Initialise using the output of Network0
prediction[:window_size] = first_sequence_coeff[:window_size]

# # OR initialise assuming that we know the first sequence
# prediction[:window_size] = ((pd.DataFrame(label[:window_size]) - mean_label)/std_label).to_numpy()

# # OR initialise with random values
# prediction[:window_size] = np.random.rand(window_size, num_feature)


# Prediction of futher label using previous steps
for ii in np.arange(window_size, len(prediction), 1):
    prediction[ii, :] = model_network1.predict( [ prediction[ii-window_size:ii, :].reshape((1, window_size, num_label)), 
                                                (feature.to_numpy())[ii-window_size:ii, :].reshape(1, window_size, 1) ],
                                               verbose=0 )


prediction = pd.DataFrame(prediction)
prediction = prediction*std_label + mean_label

toc = time.time()-tic

#%%
# plot and calculate error
error_all = np.zeros(num_label)
error_val = np.zeros(num_label)

t = 0.005*np.arange(len(prediction))

for mode in np.arange(1, num_label+1, 1):

    predict_value = np.array(prediction.iloc[:, mode-1])
    true_value = np.array(label.iloc[:, mode-1])
    
    ### calculate error for the entire dataset (train + val + test) normalised by mean absolute
    true_value_ave_abs = np.mean( np.abs(true_value) )
    error_all[mode-1] = 100*(np.mean( abs(predict_value - true_value) )/true_value_ave_abs )
    
    # a-t plot
    plt.rcParams["font.size"] = 24
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(8, 5.5))
    
    plt.plot(t, true_value, color='r', linewidth=3, label='CFD values')
    plt.plot(t, predict_value, color='b', linestyle="-.", linewidth=3.5, label='Prediction')
    # plt.plot((index_val)*0.005, true_value[index_val], linestyle="", marker="o", markersize=10, color='dimgrey', label='Validation set location')
    
    # plt.legend(framealpha=1)
    
    plt.grid(True)
    plt.tick_params(labelsize=26)
    plt.xlabel('$t(s)$', fontsize=32, math_fontfamily='cm')
    
    # Set the y-axis to scientific notation with one decimal place
    formatter = CustomScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.grid(True)
    plt.tight_layout()  # Adjusts subplot params to give specified padding
    
    
    if mode in range(1, num_modes_v+1):
        plt.title("Error =" + str(f'{error_all[mode-1]:.2f}') + "%")
        plt.ylabel(r'$a_{%s}(t)$' % mode, fontsize=32, math_fontfamily='cm')
    else:
        plt.title("Error =" + str(f'{error_all[mode-1]:.2f}') + "%")
        plt.ylabel(r'$b_{%s}(t)$' % (mode-num_modes_v), fontsize=32, math_fontfamily='cm')

    
    
    
    # ## a-a plot
    # plt.plot(true_value, predict_value, linestyle="", marker="o", color='b')
    # plt.axline((0, 0), slope=1, linestyle="--", color='k')
    # plt.xlabel('True value')
    # plt.ylabel('Prediction')
    # plt.title("Entire cycle velocity \n Mode " + str(mode) + "\n Error =" + str(f'{error_all[mode-1]:.2f}') + "%")
    # # plt.grid()
    # plt.show()


    # ### calculate error for 'test dataset' normalised by rms
    # true_value_test = true_value[index_test]
    # predict_value_test = predict_value[index_test]
    
    # true_value_test_rms = np.sqrt( np.mean( np.square(true_value_test) ) )
    # error_test[mode-1] = np.mean( abs(predict_value_test - true_value_test)/true_value_test_rms )*100
    
    # # a-a plot
    # plt.plot(true_value_test, predict_value_test, linestyle="", marker="o", color='b')
    # plt.axline((0, 0), slope=1, linestyle="--", color='k')
    # plt.xlabel('True value')
    # plt.ylabel('Prediction')
    # if mode in range(1, num_modes_v+1):
    #     plt.title("Test dataset velocity \n Mode " + str(mode) + "\n Error =" + str(f'{error_test[mode-1]:.3f}') + "%")
    # else:
    #     plt.title("Test dataset pressure \n Mode " + str(mode-num_modes_v) + "\n Error =" + str(f'{error_test[mode-1]:.3f}') + "%")
    # plt.show()


#%% save prediction of POD coeff
# np.savez('prediction_of_coeff_3DIVP_FL.npz', prediction=prediction.to_numpy())

# the next python script to use is "Analyse_and_prepare_fore_visualise"


#%%

# # Create legend elements
# cfd_line = mlines.Line2D([], [], color='r', linewidth=3, label='CFD values')
# prediction_line = mlines.Line2D([], [], color='b', linestyle="-.", linewidth=3.5, label='Prediction')

# # Plot the legend
# fig, ax = plt.subplots(figsize=(4, 3))  # Adjust the figure size to fit the horizontal legend
# ax.axis('off')  # Turn off the axis
# legend = ax.legend(handles=[cfd_line, prediction_line], loc='center', framealpha=1, ncol=3)
# plt.tight_layout()  # Adjusts subplot params to give specified padding
# plt.show()



