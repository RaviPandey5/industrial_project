#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install seaborn')


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_excel(r'C:/Users/Ravi Pandey/Documents/Internship/drive-download-20240709T093617Z-001/Agrocel_Jupyter.xlsx')


# In[3]:


df


# In[4]:


df.columns


# In[5]:


df = df.rename(columns={' Date': 'Date'})


# In[6]:


from sklearn.impute import KNNImputer

# Initialize the KNN imputer
k = 5  # Number of neighbors to consider
imputer = KNNImputer(n_neighbors=k)

df_subset = df[['Sulphur Rate']]
df_imputed = imputer.fit_transform(df_subset)
df['Sulphur Rate'] = df_imputed


# In[7]:


df['Date'] = pd.to_datetime(df['Date'])


# In[8]:


df


# In[9]:


print(df.dtypes)


# In[10]:


import matplotlib.pyplot as plt
import pandas as pd

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Plot the time series
plt.figure(figsize=(30, 20))
plt.plot(df['Date'], df['Sulphur Rate'], linestyle='-', color='b')
plt.title('Sulphur Rate Over Time', fontsize=22)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Sulphur Rate', fontsize=20)
plt.grid(True)
plt.show()


# In[11]:


from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['Sulphur Rate'],  
                            model ='multiplicative', period = 365) 


# In[12]:


result.plot()


# # ARIMA Model Preprocessing

# In[13]:


#Train_test_split

spilt = (df.index < len(df)-30)
df_train = df[spilt].copy()
df_test = df[~spilt].copy()


# In[14]:


#Stationarity checking 

from statsmodels.tsa.stattools import adfuller

# Select the column 'Sulphur Rate' from the DataFrame
sulphur_rate_data = df_train['Sulphur Rate']

# Perform the Augmented Dickey-Fuller test
adf_test = adfuller(sulphur_rate_data)
print(f'p-value: {adf_test[1]}')    #This gives us a output of p-value: 0.6980113868350736
                                    #This result shows a large p-value, which means the test fails to reject the null hypothesis. So the ADF test also suggests that our time series is non-stationary.


# In[15]:


#First Differencing

df_train_diff = df_train['Sulphur Rate'].diff().dropna()
df_train_diff.plot()


# In[16]:


adf_test = adfuller(df_train_diff)
print(f'p-value: {adf_test[1]}') # This gives us a p-value:2.1229061985843574e-16  which is less than 0.05, hence we can accept that it is stationary.


# In[17]:


#Finding the parameters p and q

from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
series = df_train_diff #insert data here
plot_acf(series) #ACF plot function
pyplot.show() #Show graph


# In[18]:


from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
series = df_train_diff #insert data here
plot_pacf(series) #ACF plot function
pyplot.show() #Show graph


# # Auto ARIMA

# In[20]:


get_ipython().system('pip install pmdarima')


# In[21]:


from pmdarima import auto_arima


model = auto_arima(df_train['Sulphur Rate'], seasonal=True, m=7, max_p=5, max_d=2, max_q=5, max_P=2, max_D=1, max_Q=2, information_criterion='aic')
print(model.summary())


# In[22]:


from sklearn.metrics import mean_squared_error, r2_score

n_forecast = len(df_test)  # Number of periods to forecast into the future
forecast = model.predict(n_periods=n_forecast)

# Step 9: Make Predictions
y_pred_test = forecast  # No need to call .values

test_score = r2_score(df_test['Sulphur Rate'], y_pred_test) 
print("Test Score (R-squared)(ARIMA Model):", test_score)


# Step 10: Evaluate the Model
mse = mean_squared_error(df_test['Sulphur Rate'], y_pred_test)
rmse = np.sqrt(mse)
print("Root Mean Squared Error(ARIMA Model):", rmse)

# Print the forecasts
print("Forecasted values:")
print(y_pred_test)


# In[23]:


plt.figure(figsize=(10, 6))
plt.plot(df_test['Date'], df_test['Sulphur Rate'], label='Actual', color='blue')
plt.plot(df_test['Date'], y_pred_test, label='Predicted', color='red')
plt.title('Actual vs Predicted Sulphur Rate(Testing Data)')
plt.xlabel('Date')
plt.ylabel('Sulphur Rate')
plt.legend()


# In[24]:


forecast = model.predict(n_periods=len(df_train))
y_pred_train = forecast


plt.figure(figsize=(10, 6))
plt.plot(df_train['Date'], df_train['Sulphur Rate'], label='Actual', color='blue')
plt.plot(df_train['Date'], y_pred_train, label='Predicted', color='red')
plt.title('Actual vs Predicted Sulphur Rate(Training Data)')
plt.xlabel('Date')
plt.ylabel('Sulphur Rate')
plt.legend()


# In[25]:


#Generate forecasts for the next 20 days
forecast_next_20_days = model.predict(n_periods=20)

# Print the forecasted values
print("Forecast for the next 20 days:")
print(forecast_next_20_days)


# In[26]:


# Plot the forecasted values along with the existing data
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Sulphur Rate'], label='Historical Data', color='blue')
plt.plot(pd.date_range(df['Date'].iloc[-1], periods=20), forecast_next_20_days, label='Forecast', color='red')
plt.title('Sulphur Rate Forecast for the Next 20 Days')
plt.xlabel('Date')
plt.ylabel('Sulphur Rate')
plt.legend()
plt.grid(True)
plt.show()


# # Manual ARIMA
# 

# In[27]:


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df_train['Sulphur Rate'], order=(2,1,1))
model_fit = model.fit()
print(model_fit.summary())


# In[28]:


from sklearn.metrics import mean_squared_error, r2_score

# Step 8: Validate the Model
# Forecast
forecast = model_fit.forecast(steps=len(df_test))

# Step 9: Make Predictions
predictions = forecast.values

# Step 10: Evaluate the Model
mse = mean_squared_error(df_test['Sulphur Rate'], predictions)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)


# In[29]:


forecast = model_fit.forecast(steps=len(df_train))
predictions = forecast.values

y_true = df_train['Sulphur Rate'].values
y_pred = forecast.values

# Calculate R-squared score
test_score = r2_score(y_true, y_pred) 
print("Test Score (R-squared):", test_score)

print("Predicted values:",y_pred)
print("True values:",y_true)


# In[31]:


from sklearn.metrics import r2_score

forecast = model_fit.forecast(steps=len(df_test))
predictions = forecast.values

# Assuming 'Sulphur Rate' is the target variable
y_true = df_test['Sulphur Rate'].values
y_pred = forecast.values

# Calculate R-squared score
test_score = r2_score(y_true, y_pred) 
print("Test Score (R-squared):", test_score)

print("Predicted values:",y_pred)
print("True values:",y_true)


# In[33]:


forecast_next_20_days = model_fit.forecast(steps=20)

# Print the forecasted values
print("Forecast for the next 20 days:")
print(forecast_next_20_days)


# In[34]:


# Plot the forecasted values along with the existing data
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Sulphur Rate'], label='Historical Data', color='blue')
plt.plot(pd.date_range(df['Date'].iloc[-1], periods=20), forecast_next_20_days, label='Forecast', color='red')
plt.title('Sulphur Rate Forecast for the Next 5 Days')
plt.xlabel('Date')
plt.ylabel('Sulphur Rate')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# # VAR (Vector Autoregression Model)

# In[35]:


df1 = pd.read_excel(r'C:/Users/Ravi Pandey/Documents/Internship/drive-download-20240709T093617Z-001/Agrocel_Jupyter.xlsx')


# In[36]:


df1


# In[37]:


df1.info()


# In[38]:


from sklearn.impute import KNNImputer

# Initialize the KNN imputer
k = 5  # Number of neighbors to consider
imputer = KNNImputer(n_neighbors=k)

df_subset = df1[['Sulphur Rate']]
df_imputed_sulphur = imputer.fit_transform(df_subset)
df1['Sulphur Rate'] = df_imputed_sulphur


# In[39]:


df1


# In[40]:


# Initialize the KNN imputer
k = 30  # Number of neighbors to consider
imputer = KNNImputer(n_neighbors=k)

df_subset = df1[['Sulphuric acid Rate']]
df_imputed_sulphuric  = imputer.fit_transform(df_subset)
df1['Sulphuric acid Rate'] = df_imputed_sulphuric


# In[41]:


df1.info()


# In[42]:


#Stationarity checking 

from statsmodels.tsa.stattools import adfuller

# Select the column 'Sulphur Rate' from the DataFrame
sulphur_rate_data = df1['Sulphur Rate']

# Perform the Augmented Dickey-Fuller test
adf_test = adfuller(sulphur_rate_data)
print(f'p-value: {adf_test[1]}')    #This gives us a output of p-value: 0.6980113868350736
                                    #This result shows a large p-value, which means the test fails to reject the null hypothesis. So the ADF test also suggests that our time series is non-stationary.


# In[43]:


#Stationarity checking 

from statsmodels.tsa.stattools import adfuller

# Select the column 'Sulphur acid Rate' from the DataFrame
sulphur_rate_data = df1['Sulphuric acid Rate']

# Perform the Augmented Dickey-Fuller test
adf_test = adfuller(sulphur_rate_data)
print(f'p-value: {adf_test[1]}')    #This gives us a output of p-value: 0.6980113868350736
                                    #This result shows a large p-value, which means the test fails to reject the null hypothesis. So the ADF test also suggests that our time series is non-stationary.


# In[44]:


#First Differencing

df_diff = df1['Sulphur Rate'].diff().dropna()
df_diff.plot()


# In[45]:


#Stationarity checking 

from statsmodels.tsa.stattools import adfuller

# Select the column 'Sulphur Rate' from the DataFrame
sulphur_rate_data = df_diff

# Perform the Augmented Dickey-Fuller test
adf_test = adfuller(sulphur_rate_data)
print(f'p-value: {adf_test[1]}')    #This gives us a output of p-value: 0.6980113868350736
                                    #This result shows a large p-value, which means the test fails to reject the null hypothesis. So the ADF test also suggests that our time series is non-stationary.


# In[46]:


from sklearn.model_selection import train_test_split
from statsmodels.tsa.api import VAR

# Step 1: Check Stationarity
acid_rate_stationary = adfuller(df1['Sulphuric acid Rate'])[1] < 0.05

# Step 2: Make "Sulphur Rate" Data Stationary
if not acid_rate_stationary:
    df['Sulphur Rate Diff'] = df1['Sulphur Rate'].diff().dropna()

# Step 3: Split the Data into Training and Testing Sets
train_df, test_df = train_test_split(df1, test_size=0.2, shuffle=False)  # Adjust test_size as needed

# Step 4: Fit VAR Model on the Training Set
if acid_rate_stationary:
    model = VAR(train_df[['Sulphur Rate', 'Sulphuric acid Rate']])
else:
    model = VAR(train_df[['Sulphur Rate Diff', 'Sulphuric acid Rate']])

results = model.fit()


# In[47]:


results.summary()


# In[ ]:




