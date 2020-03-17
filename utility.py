import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

url_total = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'
url_recovs = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'


def covid_country(country='Total', url=url_total):
  df = pd.read_csv(url)
  df.rename(
      columns={
          "Country/Region": "Country", "Province/State": "Province"
          },
      inplace=True
      )
  if country == 'Total':
    return df
  else:
    df_country = df[df['Country'].str.match(country)]
    return df_country


def get_cases(df_country, period='1/22/20'):
  return df_country.loc[:, '1/22/20':].values[0]


def GetTrainValues(dataset):
  # dataset = covid_country(country)
  '''
    Simple manipulations to have a correct dataformat for
    the training
    '''
  datasetGrouped = dataset.groupby(['Country']).sum()
  date = datasetGrouped.columns[2:]
  datasetTranspose = datasetGrouped.transpose().iloc[2:, :]
  datasetTranspose.columns = ['Cases']
  values = datasetTranspose.values

  return values, datasetTranspose, date,


def univariate_data(
    dataset, start_index, end_index, history_size, target_size
    ):
  '''
    Create the proper dataformat to make time series forecast
    - dataset: input dataset from which extract dataformats
    - start_index: starting index to be considered
    - end_index: last index of the time series to be considered
    - history_size: number of point to base the train on
    - target_size: number of future point to be predicted
    '''
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i - history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i + target_size])
  return np.array(data), np.array(labels)


def create_train_val(
    data, start_index, stop_index, past_history, future_target
    ):
  '''
    Create train and validation dataset
    -Start index: Defines the starting index that we want to consider
    -Stop index: Defines the last index that we want to consider
    -Past history: number of point to base the train on. 
    -Future target: Number of points to be predicted
    '''
  x_train_uni, y_train_uni = univariate_data(
      data, start_index, stop_index, past_history, future_target
      )
  x_val_uni, y_val_uni = univariate_data(
      data, start_index, None, past_history, future_target
      )

  return (x_train_uni, y_train_uni), (x_val_uni, y_val_uni)


def create_time_steps(length):
  return list(range(-length, 0))


def show_plot(plot_data, delta, title):
  '''
    Plot history series with target and the predicted targe
    - plot_data: [x_train, y_train]
    - delta: distance between history series and future target
    - title: title for the image
    '''

  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future + 5) * 2])
  plt.xlabel('Time-Step')
  plt.show()
  # return plt


def plot_losses(history, file_name):
  '''
    Plot loss functions, both train and validation
    - history: history object obtained from model.fit()
    - file_name: output path for save
    '''
  train_loss = history.history['loss']
  plt.plot(train_loss, label='Train')
  validation_loss = history.history['val_loss']
  plt.plot(validation_loss, label='Validation')
  plt.title("Loss functions")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  # plt.show()
  plt.savefig('LSTM/plot/' + file_name)


def generate_saveString(model, epochs):
  ''' 
    Generate string from model with model parameter
    - Loss function
    - Number of epochs
    TODO:
    Add more info!
    '''
  string = 'model_'
  string = string + model.loss + '_' + str(epochs)
  return string
