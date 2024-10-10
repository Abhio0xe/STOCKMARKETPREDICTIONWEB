import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model('/Users/abhishekmishra/Documents/AIML_Project/Stock_Trend_prediction_web_app_python/Stock_Prediction_Model.keras')
# Header of the web app using Streamlit
st.header('Stock Market Predictor')

# Input for the stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Time range for data
start = '2013-10-20'
end = '2023-10-20'

# Download stock data using yfinance
data = yf.download(stock, start=start, end=end)

# Display the stock data
st.subheader('Stock Data')
st.write(data)

# Train-test split (80% train, 20% test)
data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

# MinMaxScaler for normalization
scaler = MinMaxScaler(feature_range=(0,1))

# Prepare the last 100 days of training data to add to the test set
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)

# Scale the test data
data_test_scaled = scaler.fit_transform(data_test)

# Prepare the x and y for test data
x_test = []
y_test = []

for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Now you can use the `x_test` and `y_test` for predictions or further processing.