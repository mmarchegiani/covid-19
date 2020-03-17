import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
# import plotly.express as px
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import io
import requests
from matplotlib.colors import LogNorm
from utility import *
import tensorflow as tf

matplotlib.rcParams['figure.figsize'] = [15, 10]
matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15

model_path = 'LSTM/models/'
train_plot_path = 'LSTM/plot/'

#Get dataset
country = 'China'
dataset = covid_country(country)

#Get active cases and date
cases, datasetTranspose, date = GetTrainValues(dataset)

#Get resampled data
rule = '1D'    #upsampling

if ("." in rule):
  raise Exception("Dot not allowed, change with integer!")
#keep linear for now as method
resampled, series = upsamplingData(dataset, rule=rule, method='linear')

#Quick plot to see the behaviour
datasetTranspose.plot(subplots=True)
plt.title(country + ' cases')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()

#Preparing for the training, starting with scaling inputs

scaler = MinMaxScaler()
scaled = scaler.fit_transform(resampled)

#Create train and validation dataformats
#Past history, 60% of the total
past_history = int(len(scaled) * 0.6)
#... the remaining have to be predicted
future_target = int(len(scaled) * 0.25)

end_index = len(scaled) - future_target
xyTrain, xyVal = create_train_val(scaled, 0, end_index, past_history, future_target)
x_train_uni, y_train_uni = xyTrain[0], xyTrain[1]
x_val_uni, y_val_uni = xyVal[0], xyVal[1]
# print("Train \n {} \n Validation \n {} \n".format(xyTrain, xyVal))
# print(xyTrain[0])

show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')


def baseline(history):
  return np.mean(history)


show_plot(
    [x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])],
    future_target,
    'Baseline Prediction (Based on mean)'
    )

#TODO: pass train parameters from command line

BATCH_SIZE = 1
BUFFER_SIZE = 30

train_univariate = tf.data.Dataset.from_tensor_slices(
    (x_train_uni, y_train_uni)
    )
train_univariate = train_univariate.cache(
).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# for x,y in train_univariate.take(1):
#     print(x,y)

# for element in train_univariate:
#     print(element)

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

# for element in val_univariate:
#     print(element)

activation = 'relu'

# NOTE: ERROR USING TANH ACTIVATION FUNCTION, TENSORFLOW ISSUE https://github.com/tensorflow/tensorflow/issues/30263
#Let's stay with relu or sigmoid for now

if (activation == 'tanh'):
  raise Exception('tanh has issue, please change activation function!')

lstm_model = tf.keras.models.Sequential(
    [
        tf.keras.layers.LSTM(
            1,
            activation=activation,
            input_shape=x_train_uni.shape[-2:],
            return_sequences=True
            ),
        tf.keras.layers.Dense(1)
        ]
    )

optimizer = ['adam']
loss_function = ['mse', 'binary_crossentropy']

lstm_model.compile(optimizer=optimizer[0], loss=loss_function[0])

print("\n\n {}".format(lstm_model.summary()))

print("@@@@@ Train prediction shape @@@@@ \n")

for x, y in val_univariate.take(1):
  print(lstm_model.predict(x).shape)

EVALUATION_INTERVAL = 100
EPOCHS = 10
VALIDATION_STEP = 50

print("@@@@@ Starting the training  @@@@@ \n")
history = lstm_model.fit(
    train_univariate,
    epochs=EPOCHS,
    steps_per_epoch=EVALUATION_INTERVAL,
    validation_data=val_univariate,
    validation_steps=VALIDATION_STEP
    )

file_name = generate_saveString(lstm_model, EPOCHS)

if (rule != '1D'):
  file_name = file_name + '_' + rule
lstm_model.save(model_path + file_name + '.h5')
plot_losses(history, file_name + '.png')
