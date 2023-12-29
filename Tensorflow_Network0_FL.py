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

# flow field 1: -25%
temp = np.load('../snapshot_matrix_CFD_minus25_FL.npz')
X_1, X_mean_1, Xp_1, Xp_mean_1 = ( temp['X'], temp['X_mean'], temp['Xp'], temp['Xp_mean'] )
X_1 = X_1 + X_mean_1
Xp_1 = Xp_1 + Xp_mean_1
del temp, X_mean_1, Xp_mean_1

temp = np.load('../InletFLowRate_minus25.npz')
flowrate_1 = temp['flowrate']
del temp

# flow field 2: +25%
temp = np.load('../snapshot_matrix_CFD_plus25_FL.npz')
X_2, X_mean_2, Xp_2, Xp_mean_2 = ( temp['X'], temp['X_mean'], temp['Xp'], temp['Xp_mean'] )
X_2 = X_2 + X_mean_2
Xp_2 = Xp_2 + Xp_mean_2
del temp, X_mean_2, Xp_mean_2

temp = np.load('../InletFLowRate_plus25.npz')
flowrate_2 = temp['flowrate']
del temp


# X_mean, Xp_mean
temp = np.load('snapshot_matrix_traning_minus25_plus25_FL.npz')
X_mean, Xp_mean = ( temp['X_mean'], temp['Xp_mean'] )
del temp

X_1 = X_1 - X_mean
Xp_1 = Xp_1 - Xp_mean

X_2 = X_2 - X_mean
Xp_2 = Xp_2 - Xp_mean

# POD modes (basis vectors)
temp = np.load('POD_traning_minus25_plus25_FL.npz')
U, Up = ( temp['U'], temp['U_p'] )
del temp

coeff_1 = U.T @ X_1
coeff_p_1 = Up.T @ Xp_1

coeff_2 = U.T @ X_2
coeff_p_2 = Up.T @ Xp_2


#%% define the datasets

nt = np.min( [coeff_1.shape[1], coeff_2.shape[1]] ) # number of temporal points per flow field

# making them equal in length (nt) makes our life easy
flowrate_1 = flowrate_1[: nt]
coeff_1 = coeff_1[:, : nt]
coeff_p_1 = coeff_p_1[:, : nt]

flowrate_2 = flowrate_2[: nt]
coeff_2 = coeff_2[:, : nt]
coeff_p_2 = coeff_p_2[:, : nt]


# we should merge the data to calculate mean ans std of the whole data

# feature
feature = pd.DataFrame( np.vstack( (flowrate_1, flowrate_2) ) )
num_feature = len(feature.columns)

# label
num_modes_v = 22 # number of velocity POD modes included we want to predict (for the reconstruction)
num_modes_p = 2 # number of pressure POD modes included we want to predict (for the reconstruction)
num_label = num_modes_v + num_modes_p

label_velocity = np.transpose( np.hstack( (coeff_1[0: num_modes_v, 0: nt], coeff_2[0: num_modes_v, 0: nt]) ) )
label_velocity = pd.DataFrame(label_velocity)

label_pressure = np.transpose( np.hstack( (coeff_p_1[0: num_modes_p, 0: nt], coeff_p_2[0: num_modes_p, 0: nt]) ) )
label_pressure = pd.DataFrame(label_pressure)

label = pd.concat([label_velocity, label_pressure], axis=1)
label.columns = range(label.columns.size) # reset column names to numbers


# standardise the data
mean_feature = feature.mean()
std_feature = feature.std()
feature = (feature - mean_feature)/std_feature

mean_label = label.mean()
std_label = label.std()
label = (label - mean_label)/std_label

# %% save mean and std of feature and label
# np.savez('mean_and_std_for_Network0_and1.npz', mean_feature=mean_feature, std_feature=std_feature, mean_label=mean_label, std_label=std_label)

#%% put data into window format

window_size = 4
batch_size = 1 # it's equal to 1 in this step

# we need to separate again before make window dataset (otherwise there will be mix window which is wrong)
feature_1 = feature[: nt]
feature_2 = feature[nt: ]

label_1 = label[: nt]
label_2 = label[nt: ]

# now we can arrange them to window format
feature_window_1 = sliding_windows(feature_1, window_size)
feature_window_2 = sliding_windows(feature_2, window_size)

label_window_1 = sliding_windows(label_1, window_size)
label_window_2 = sliding_windows(label_2, window_size)

# and now, we merge them back
feature_window = np.concatenate( (feature_window_1, feature_window_2), axis=0 )
label_window = np.concatenate( (label_window_1, label_window_2), axis=0 )


# split the data
split_ratio = 0.10
index_val = np.random.randint(0, len(feature_window), int(split_ratio*len(feature_window)))
index_train = np.setdiff1d(np.arange(0, len(feature_window), 1), index_val)

feature_window_train = feature_window[index_train]
feature_window_val = feature_window[index_val]

label_window_train = label_window[index_train]
label_window_val = label_window[index_val]

#%% model

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(200, activation='tanh', input_shape=(window_size, num_feature), return_sequences=True),
    tf.keras.layers.LSTM(200, activation='tanh', return_sequences=False),
    tf.keras.layers.Dense(100, activation='tanh'),
    tf.keras.layers.Dense(100, activation='tanh'),
    tf.keras.layers.Dense(window_size*num_label), # output layer
    tf.keras.layers.Reshape([window_size, num_label]) # reshape the output to window format
    ]) 


# initial_learning_rate = 0.01
# final_learning_rate = 0.0001
# learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/1000)
# steps_per_epoch = int(1000/100)

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#                 initial_learning_rate=initial_learning_rate,
#                 decay_steps=steps_per_epoch,
#                 decay_rate=learning_rate_decay_factor,
#                 staircase=True)

model.compile(optimizer= tf.keras.optimizers.Adamax(learning_rate = 0.001),
              loss='mse',
              metrics= ['mape', 'mse'])


model.summary()

#%% Train the model
tic = time.time()

# use early_stopping to make the model automatically stop when it no longer improve
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=250, restore_best_weights=True)


# train the model
history = model.fit(x=feature_window_train, y=label_window_train, validation_data=(feature_window_val, label_window_val), epochs=3000, verbose=1, 
                    callbacks=[early_stopping], 
                    shuffle=True, batch_size=32)

toc = time.time()-tic
print("total time used to train: ", round(toc, 1), "s")
#%% Plot learning curves

loss     = history.history['mse']
val_loss = history.history['val_mse']
epochs   = np.arange(1, len(loss)+1, 1) # Get number of epochs

plt.plot(epochs, loss, label='training')
plt.plot(epochs, val_loss, label='validation')
plt.title('Training loss')
plt.legend()
# plt.xlim([0, 500])
# plt.ylim([0, 10])
plt.figure()  


#%% Evaluate your model on the test set
model.evaluate(feature_window_val, label_window_val)

#%%

# model.save('Network0.h5')

#%%

# model = load_model('Network0.h5')

#%% Get prediction values

# evaluate the network on one of the training flow field
feature_window_for_prediction = feature_window_1

label_for_plot = pd.DataFrame(label[:nt])
label_for_plot = label_for_plot*std_label + mean_label

index_val_for_plot = index_val[index_val <= nt]
# index_val_for_plot = index_val[index_val >= nt] -nt


prediction_window = model.predict(feature_window_for_prediction, verbose=0)  # Prediction of feature_window

prediction = np.zeros(np.shape(label_for_plot))
prediction[0: window_size] = prediction_window[0]
for ii in np.arange(window_size, len(prediction_window), 1):
    prediction[ii, :] = prediction_window[ii-window_size+1, window_size-1, :]
    

prediction = pd.DataFrame(prediction)
prediction = prediction*std_label + mean_label

#%%
t = 0.005*np.arange(label_for_plot.shape[0])

# plot and calculate error
error_all = np.zeros(num_label)
error_val = np.zeros(num_label)
for mode in np.arange(1, num_label+1, 1):
    
    true_value = np.array(label_for_plot.iloc[:, mode-1])
    predict_value = np.array(prediction.iloc[:, mode-1])
    
    ### calculate error for the entire dataset (train + val + test) normalised by average absolute value
    true_value_ave_abs = np.mean( np.abs(true_value) )
    error_all[mode-1] = 100*(np.mean( abs(predict_value - true_value) )/true_value_ave_abs )
    
    # a-t plot
    plt.rcParams["font.size"] = 24
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(8, 5.5))
    
    plt.plot(t, true_value, color='r', linewidth=3, label='CFD values')
    plt.plot(t, predict_value, color='b', linestyle="-.", linewidth=3.5, label='Prediction')
    plt.plot((index_val_for_plot)*0.005, true_value[index_val_for_plot], linestyle="", marker="o", markersize=10, color='dimgrey', label='Validation set location')
    
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


# %% save prediction of POD coeff
# np.savez('prediction_of_coeff_3DCFD.npz', prediction=prediction.to_numpy())


