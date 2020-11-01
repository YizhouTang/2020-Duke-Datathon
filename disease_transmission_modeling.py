import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

#Load data
df = pd.read_csv('covid_cases.csv',index_col = "Date")#,parse_dates = True,index_col='date')

county = "Kings New York"
df_county = df[[county+" Cases",county+" Deaths"]]

df_confirmed = df_county[county+" Cases"]
print("df_confirmed")
print(df_confirmed.head())

df_dead = df_county[county+ " Deaths"]
print("df_dead")
print(df_dead)

df_county.plot(title = county + " COVID-19 Stats")

#Train test split
split = int(len(df_confirmed)*2/3)
train = np.array(df_confirmed.iloc[:split]).reshape(-1, 1)
test = np.array(df_confirmed.iloc[split:]).reshape(-1,1)

#Normalize
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


n_input = 5  ## number of steps
n_features = 1 ## number of features you want to predict (for univariate time series n_features=1)
# generator = TimeseriesGenerator(scaled_train,scaled_train,length = n_input,batch_size=1)
#
# print("Scaled Train")
# print(len(scaled_train))
# print("Generator")
# print(len(generator))
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
