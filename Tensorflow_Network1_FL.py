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

# feature and labels
num_modes_v = 22 # number of velocity POD modes included we want to predict (for the reconstruction)
num_modes_p = 2 # number of pressure POD modes included we want to predict (for the reconstruction)
num_label = num_modes_v + num_modes_p
num_feature = num_label

label_velocity = np.transpose( np.hstack( (coeff_1[0: num_modes_v, 0: nt], coeff_2[0: num_modes_v, 0: nt]) ) )
label_velocity = pd.DataFrame(label_velocity)

label_pressure = np.transpose( np.hstack( (coeff_p_1[0: num_modes_p, 0: nt], coeff_p_2[0: num_modes_p, 0: nt]) ) )
label_pressure = pd.DataFrame(label_pressure)

label = pd.concat([label_velocity, label_pressure], axis=1)
label.columns = range(label.columns.size) # reset column names to numbers


# standardise the data
mean_label = label.mean()
std_label = label.std()

label = (label - mean_label)/std_label


# feature branch B (flowrate)
feature_B = pd.DataFrame( np.vstack( (flowrate_1, flowrate_2) ) )

num_feature_B = len(feature_B.columns)

# standardise the data B
mean_feature_B = feature_B.mean()
std_feature_B = feature_B.std()
feature_B = (feature_B - mean_feature_B)/std_feature_B


#%% put data into window format

window_size = 4

# we need to separate again before make window dataset (otherwise there will be mix window which is wrong)
feature_B_1 = feature_B[: nt]
feature_B_2 = feature_B[nt: ]

label_1 = label[: nt]
label_2 = label[nt: ]


# now we can arrange them to window format
feature_window_1 = sliding_windows(label_1, window_size)
feature_window_2 = sliding_windows(label_2, window_size)

feature_B_window_1 = sliding_windows(feature_B_1, window_size)
feature_B_window_2 = sliding_windows(feature_B_2, window_size)

label_window_1 = label_1[window_size:].to_numpy()
label_window_2 = label_2[window_size:].to_numpy()


# and now, we merge them back
feature_window = np.concatenate( (feature_window_1, feature_window_2), axis=0 )
feature_B_window = np.concatenate( (feature_B_window_1, feature_B_window_2), axis=0 )
label_window = np.concatenate( (label_window_1, label_window_2), axis=0 )



# split the data
split_ratio = 0.1
index_val = np.random.randint(0, len(feature_window), int(split_ratio*len(feature_window)))
index_train = np.setdiff1d(np.arange(0, len(feature_window), 1), index_val)

feature_A_window_train = feature_window[index_train]
feature_B_window_train = feature_B_window[index_train]
label_window_train = label_window[index_train]

feature_A_window_val = feature_window[index_val]
feature_B_window_val = feature_B_window[index_val]
label_window_val = label_window[index_val]

#%% model

# define two sets of inputs
inputA = Input(shape=(window_size, num_feature))  # POD coeffs
inputB = Input(shape=(window_size, 1)) # flowrate

# block 2 (branch A) (POD coeff branch)
x = LSTM(64, activation='tanh', return_sequences=True)(inputA)
x = LSTM(64, activation='tanh', return_sequences=False)(x)
x = Dense(32, activation='tanh')(x)
x = Model(inputs=inputA, outputs=x)

# block 3 (branch B) (flowrate branch)
y = LSTM(64, activation='tanh', return_sequences=True)(inputB)
y = LSTM(64, activation='tanh', return_sequences=False)(y)
y = Dense(10, activation='tanh')(y)
y = Model(inputs=inputB, outputs=y)

# combine the output of the two branches
combined = tf.keras.layers.concatenate([x.output, y.output])

# apply a FC layer and then a regression prediction on the
# block 4 (combined outputs)
z = Dense(64, activation="tanh")(combined)
z = Dense(64, activation="tanh")(z)
z = Dense(num_label)(z)

# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[x.input, y.input], outputs=z)

model.compile(optimizer= tf.keras.optimizers.Adamax(learning_rate = 0.003),
              loss='mse',
              metrics= ['mape', 'mse'])

model.summary()

#%% Train the model
tic = time.time()

# use early_stopping to make the model automatically stop when it no longer improve
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=80, restore_best_weights=True)

# train the model
history = model.fit(x=[feature_A_window_train, feature_B_window_train], 
                    y=label_window_train,
                    validation_data=([feature_A_window_val, feature_B_window_val], label_window_val),
                    epochs=1500, verbose=1, 
                    callbacks=[early_stopping], shuffle=True, batch_size=32)

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
# plt.xlim([0, 100])
# plt.ylim([0, 0.01])
plt.figure()  


#%% Evaluate your model on the test set
model.evaluate(x=[feature_A_window_train, feature_B_window_train], 
                y=label_window_train,)

#%%

# model.save('Network1.h5')

#%%

# model = load_model('Network1.h5')

#%% Get prediction values

# evaluate the network on one of the training flow field
feature_window_for_prediction = feature_window_2
feature_B_window_for_prediction = feature_B_window_2

label_for_plot = label_2
label_for_plot = label_for_plot*std_label + mean_label

# index_val_for_plot = index_val[index_val < nt]
index_val_for_plot = index_val[index_val > nt] -nt


# Initialise using real values
prediction = np.zeros(label_for_plot.shape)
prediction[:window_size] = (label_for_plot[:window_size] - mean_label)/std_label

# Prediction of futher label using previous steps
for ii in np.arange(window_size, len(prediction), 1):
    prediction[ii, :] = model.predict( [ prediction[ii-window_size:ii, :].reshape((1, window_size, num_label)), 
                                        feature_B_window_for_prediction[ii-window_size, :].reshape(1, window_size, 1) ],
                                        verbose=0 )

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
    
    ### calculate error for the entire dataset (train + val + test) normalised by mean absolute
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


#%% save prediction of POD coeff
# np.savez('prediction_of_coeff_3DCFD.npz', prediction=prediction.to_numpy())

#%%

# # Create legend elements
# cfd_line = mlines.Line2D([], [], color='r', linewidth=3, label='CFD values')
# prediction_line = mlines.Line2D([], [], color='b', linestyle="-.", linewidth=3.5, label='Prediction')
# validation_set_marker = mlines.Line2D([], [], color='dimgrey', marker="o", markersize=10, linestyle='', label='Validation set location')

# # Plot the legend
# fig, ax = plt.subplots(figsize=(4, 3))  # Adjust the figure size to fit the horizontal legend
# ax.axis('off')  # Turn off the axis
# legend = ax.legend(handles=[cfd_line, prediction_line, validation_set_marker], loc='center', framealpha=1, ncol=3)
# plt.tight_layout()  # Adjusts subplot params to give specified padding
# plt.show()

#%%

# # Setup
# plt.rcParams["font.size"] = 24
# plt.rcParams["font.family"] = "Arial"
# plt.figure(figsize=(12, 8))

# # Sample data (assuming flowrate and t are defined in your original code)
# t = np.arange(1, len(flowrate)+1, 1)*0.005

# # Your plotting code
# plt.plot(t, flowrate, color='k', linewidth=3)
# # plt.legend(framealpha=1)
# plt.grid(True)
# plt.tick_params(labelsize=26)
# plt.xlabel('$t(s)$', fontsize=32, math_fontfamily='cm')
# plt.ylabel('Inlet mass flow rate ' '$(kg/s)$', fontsize=32, math_fontfamily='cm')

# # Custom tick formatter function
# def format_ticks(x, pos):
#     return f'{x * 1e3:.2f}'

# # Set y-axis formatter to custom function
# y_formatter = FuncFormatter(format_ticks)
# plt.gca().yaxis.set_major_formatter(y_formatter)

# # Add the multiplier at the top of the y-axis
# plt.gca().annotate(r'$\times 10^3$', xy=(0.0, 1.01), xycoords='axes fraction',
#                    fontsize=24, ha='left', va='bottom')

# plt.grid(True)
# plt.tight_layout()
# plt.show()
