#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install mplfinance


# In[5]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import math
import random
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.preprocessing import MinMaxScaler
from tensorflow. keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error


# In[9]:


df = pd.read_csv(r"C:\Users\Nischala\OneDrive\Desktop\Datasets\CAC40_stocks_2010_2021.csv",parse_dates=['Date'])
df.head()


# In[10]:


def specific_data(company, start, end):
    company_data = df [df ['StockName'] == company]
    date_filtered_data = company_data[(company_data['Date'] > start) & (company_data['Date'] < end)]
    return date_filtered_data


# In[17]:


company_name = random.choice (df ['StockName'].unique().tolist())
start_date=dt.datetime (2014,1,1)
end_date=dt.datetime (2020,1,1)
specific_df = specific_data (company_name, start_date, end_date)


# In[18]:


specific_df.head()


# In[19]:


specific_df ['Date'] = pd.to_datetime (specific_df ['Date'])
plt.figure(figsize=(15, 6))
plt.plot(specific_df ['Date'], specific_df ['Close'], marker='.')
plt.title('Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[20]:


matplotlib_date = mdates.date2num (specific_df ['Date'])
ohlc = np.vstack ( (matplotlib_date, specific_df ['Open'], specific_df ['High'], specific_df ['Low'], specific_df ['Close'])).T
plt.figure(figsize=(15, 6))
ax = plt.subplot()
candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r')
ax.xaxis_date()
plt.title('Candlestick Chart')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[21]:


window = 30
plt.figure(figsize=(15, 6))
plt.plot(specific_df ['Date'], specific_df ['Close'], label='Closing Price', linewidth=2)
plt.plot(specific_df ['Date'], specific_df ['Close']. rolling (window=window).mean(), label=f' {window}-Day Moving Avg', linestyle='--')
plt.title(f'Closing Prices and {window}-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt. legend()
plt.grid(True)
plt.show()


# In[22]:


new_df = specific_df.reset_index() ['Close']


# In[23]:


scaler = MinMaxScaler()
scaled_data=scaler.fit_transform(np.array(new_df).reshape(-1,1))


# In[24]:


train_size = int(len(scaled_data) * 0.8) # 80% for training
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]


# In[26]:


n_past = 60
X_train, y_train = [], []
for i in range(n_past, len(train_data)):
    X_train.append(train_data[i- n_past:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_test, y_test = [], []
for i in range(n_past, len(test_data)):
    X_test.append(test_data[i- n_past:i, 0])
    y_test.append(test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)


# In[28]:


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[30]:


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout (0.2)) # Adding dropout to prevent overfitting
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout (0.2))
model.add(LSTM(units=50)) 
model.add(Dropout (0.2))
model.add(Dense (units=1))


# In[31]:


model.summary()


# In[32]:


model.compile(loss="mean_squared_error",optimizer='adam')


# In[34]:


# checkpoints = ModelCheckpoint (filepath = 'my_weights.h5', save_best_only = True)
early_stopping = EarlyStopping (monitor= 'val_loss', patience=15, restore_best_weights=True)
model.fit(X_train, y_train,
    validation_data= (X_test,y_test),
    epochs=100,
    batch_size=32,
    verbose=1,
         )
#     callbacks = [checkpoints, early_stopping])


# In[35]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[36]:


print(math.sqrt (mean_squared_error(y_train, train_predict)))
print(math.sqrt(mean_squared_error(y_test, test_predict)))


# In[38]:


look_back = 60

trainPredictPlot = np.empty_like(new_df)
trainPredictPlot[:] = np.nan
trainPredictPlot [look_back: len (train_predict) +look_back] = train_predict. flatten()

testPredictPlot = np.empty_like(new_df)
testPredictPlot[:] = np.nan
test_start = len(new_df) - len(test_predict)
testPredictPlot [test_start:] = test_predict. flatten()

original_scaled_data = scaler.inverse_transform(scaled_data)

plt.figure(figsize=(15, 6))
plt.plot(original_scaled_data, color='black', label=f"Actual {company_name} price")
plt.plot(trainPredictPlot, color='red', label=f"Predicted {company_name} price (train_set)")
plt.plot(testPredictPlot, color='blue', label="Predicted {company_name} price (test_set)")
plt.title(f" {company_name} share price")
plt.xlabel("time")

plt.ylabel(f"{company_name} share price")
plt. legend()
plt.show()


# In[40]:


last_sequence = X_test [-1]
last_sequence = last_sequence.reshape(1, n_past, 1)

predictions_next_10_days = []
for _ in range(10):
    next_day_prediction = model.predict(last_sequence)
    predictions_next_10_days.append(next_day_prediction [0, 0]) # Get the predicted value
    last_sequence = np.roll(last_sequence, -1, axis=1) # Shift the sequence by one day
    last_sequence [0, -1, 0] = next_day_prediction # Update the last element with the new prediction
predictions_next_10_days = scaler.inverse_transform (np.array (predictions_next_10_days).reshape(-1, 1))

print("Predictions for the next 10 days: ")

for i, prediction in enumerate (predictions_next_10_days, start=1):
    print (f"Day {i}: Predicted Price = {prediction[0]}")


# In[ ]:




