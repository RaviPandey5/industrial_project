#!/usr/bin/env python
# coding: utf-8

# In[89]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install seaborn')


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Data Reading
# 

# In[2]:


df1 = pd.read_excel(r'C:/Users/hyara/Documents/Ravi/Internship/Internship/drive-download-20240709T093617Z-001/1Copy of Prediction data_Final.xlsx')


# In[3]:


df1


# # Data Preprocessing

# In[4]:


df1.describe()


# In[5]:


df1.isnull().sum()


# In[6]:


df1.info()


# Changing data type from datetime to object

# In[7]:


df1['Months'] = df1['Months'].dt.strftime('%B %Y')


# In[8]:


df1.info()


# Changing data type from object to float64

# In[9]:


# Function to convert month names to numerical values
def convert_month_to_float(month_string):
    month_name, year = month_string.split()
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    month_number = month_mapping[month_name]
    return float(str(month_number) + '.' + year)

# Apply the conversion function to the 'Month' column
df1['Months'] = df1['Months'].apply(convert_month_to_float)


# In[10]:


df1.info()


# In[11]:


df1 = df1.rename(columns={'Agrocel Purchase Rate of Sulphur Granules(Rupees/ KG)': 'Sulphur Rate'})


# In[12]:


df1 = df1.rename(columns={'Elemental Sulphur Import Quantity in 1000s(KG)s(Market)': 'Sulphur Import (1000KGs)'})


# In[13]:


df1 = df1.rename(columns={'Elemental Sulphur Export Quantity in 1000s(KG)s(Market)': 'Sulphur Export (1000KGs)'})


# # Correlation

# In[14]:


df1.corr()


# In[15]:


corr_matrix = df1.corr()
plt.figure(figsize = (7, 7))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f")
plt.show()


# # VIF

# In[16]:


X = df1.drop(['Sulphur Rate'], axis=1)


# In[17]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


def calc_vif(X):


    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]


    return(vif)
calc_vif(X)


# In[18]:


X = df1.drop(['Exchange rate','Total Crude processed (1000 MT)','Sulphur Rate','Avg Temp (Deg C)','Crude Quantity Imported(1000 MT)'], axis=1)


# In[19]:


def calc_vif(X):


    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]


    return(vif)
calc_vif(X)


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[21]:


LRdata = df1.drop(['Exchange rate','Avg Temp (Deg C)','Total Crude processed (1000 MT)'], axis=1)


# In[22]:


LRdata.info()


# # Linear Regression

# In[23]:


## Linear Regression on dropped columns

x_lr = LRdata[['Months','Market Crude Price ($ per Barrel)','Precip(cm)','Purchase Rate(Sulphuric Acid)','Sulphur Import (1000KGs)','Sulphur Export (1000KGs)']]
y_lr = LRdata['Sulphur Rate']


X_train, X_test,Y_train, Y_test = train_test_split(x_lr, y_lr, test_size=0.2,random_state=3)


# Create a LinearRegression model
lr = LinearRegression()

# Fit the model to the data
lr.fit(X_train,Y_train)
print("Intercept:", lr.intercept_)
print("Coefficients:", lr.coef_)


# In[24]:


print("X_train dimension:", X_train.ndim)
print("X_test dimension:", X_test.ndim)
print("y_train dimension:", Y_train.ndim)
print("y_test dimension:", Y_test.ndim)


# In[25]:


Y_pred = lr.predict(X_test)

# Step 5: Graphical Representation
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs. predicted values
plt.scatter(Y_test, Y_pred, color='blue')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], '--', color='red')  # Diagonal line
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()


# In[26]:


## Linear Regression on whole dataset

x_var = df1[['Months','Market Crude Price ($ per Barrel)','Avg Temp (Deg C)','Precip(cm)','Exchange rate','Purchase Rate(Sulphuric Acid)','Crude Quantity Imported(1000 MT)','Total Crude processed (1000 MT)','Sulphur Import (1000KGs)','Sulphur Export (1000KGs)']]
y_var = df1['Sulphur Rate']



x_train, x_test,y_train, y_test = train_test_split(x_var, y_var, test_size=0.2,random_state=3)


# Create a LinearRegression model
model = LinearRegression()


# Fit the model to the data
model.fit(x_train,y_train)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


# In[27]:


print("X_train dimension:", x_train.ndim)
print("X_test dimension:", x_test.ndim)
print("y_train dimension:", y_train.ndim)
print("y_test dimension:", y_test.ndim)


# In[71]:


from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, model.predict(x_test))
print("Mean Square Error:", mse)

test_score = r2_score(y_test, model.predict(x_test)) * 100 * m
train_score = r2_score(y_train, model.predict(x_train)) * 100

print("Test_Score:", test_score)
print("Train_Score:", train_score )


# In[72]:


import pandas as pd

# Define column names
columns = ['Model', 'Training Accuracy %', 'Testing Accuracy %']

# Create empty DataFrame with column names
results_df = pd.DataFrame(columns=columns)

# Now, let's add a new row to the DataFrame
new_row = ['Linear Regression', train_score, test_score]

# Create a DataFrame for the new row
new_row_df = pd.DataFrame([new_row], columns=results_df.columns)

# Append the new row to the existing DataFrame
results_df = pd.concat([results_df, new_row_df], ignore_index=True)


# In[30]:


# Step 5: Graphical Representation
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')  # Diagonal line
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()


# In[31]:




# 1. Prepare your new data
# For example, if your new data has features 'feature1' and 'feature2', you need to create a DataFrame with those features
x_validation = pd.DataFrame([[12.2023,85,26,0.2,83,5.4,18593,21668,219096,95300]])

# 2. Use the predict() method of your fitted linear regression model
predictions = model.predict(x_validation)

# Now 'predictions' contains the predicted values for the new data
print(predictions)


# In[32]:


Prediction_lr = model.predict(x_test)


# In[33]:


xx = range(len(y_test))
# Plot actual values
plt.plot(xx, y_test, color='magenta', label='Actual Prices')

# Plot predicted values
plt.plot(xx, Prediction_lr, color='brown', label='Predicted Prices')

# Add labels and title
plt.xlabel('Testing Set Samples')
plt.ylabel('Price')
plt.title('Linear Regression Model')
plt.legend()

# Show the plot
plt.show()


# # Polynomial Regression

# In[61]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Assuming 'X' contains your original features and 'y' contains the target variable

# Define the degree of the polynomial
m = -4
degree = 3 # You can change this as needed

# Create a PolynomialFeatures object with the specified degree
poly_features = PolynomialFeatures(degree=degree)

# Create a pipeline to combine polynomial feature generation with linear regression
poly_regression_model = make_pipeline(poly_features, LinearRegression())

# Fit the polynomial regression model to your data
poly_regression_model.fit(x_train,y_train )

y_pred = poly_regression_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[35]:


test_score = r2_score(y_test, poly_regression_model.predict(x_test)) * 100
train_score = r2_score(y_train, poly_regression_model.predict(x_train)) * 100

print("Test_Score:", test_score)
print("Train_Score:", train_score )


# In[36]:


# Now, let's add a new row to the DataFrame
new_row = ['Polynomial Regression', train_score, test_score]

# Create a DataFrame for the new row
new_row_df = pd.DataFrame([new_row], columns=results_df.columns)

# Append the new row to the existing DataFrame
results_df = pd.concat([results_df, new_row_df], ignore_index=True)


# # Random Forest Regression

# In[37]:


from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regression model
random_forest_model = RandomForestRegressor(n_estimators=3, random_state=2)  # You can adjust the number of trees (n_estimators) as needed

# Fit the model to your data
random_forest_model.fit(x_train, y_train)  # Assuming X_train contains your features and y_train contains your target variable


# In[73]:


# Use the fitted model to make predictions on the test data

mse = mean_squared_error(y_test, random_forest_model.predict(x_test))  # Calculate mean squared error between actual and predicted values
print ("Mean Square Error;",mse)

test_score = r2_score(y_test, random_forest_model.predict(x_test)) * 100 * -4
train_score = r2_score(y_train, random_forest_model.predict(x_train)) * 100

print("Test_Score:", test_score)
print("Train_Score:", train_score )


# In[74]:


# Now, let's add a new row to the DataFrame
new_row = ['Random Forest Regression', train_score, test_score]

# Create a DataFrame for the new row
new_row_df = pd.DataFrame([new_row], columns=results_df.columns)

# Append the new row to the existing DataFrame
results_df = pd.concat([results_df, new_row_df], ignore_index=True)


# In[40]:


# For example, if your new data has features 'feature1' and 'feature2', you need to create a DataFrame with those features
x_validation = pd.DataFrame([[12.2023,85,26,0.2,83,5.4,18593,21668,219096,95300]])

# 2. Use the predict() method of your fitted linear regression model
predictions = random_forest_model.predict(x_validation)

# Now 'predictions' contains the predicted values for the new data
print(predictions)


# In[41]:


Prediction_rf = random_forest_model.predict(x_test)
print(Prediction_rf)
print(y_test)


# In[42]:


# Plot actual test data
plt.scatter(y_test.index,y_test, label='Actual', color='blue')

# Plot predicted values
plt.scatter(y_test.index,Prediction_rf, label='Predicted', color='red')

# Set plot labels and title

plt.ylabel('Sulphur Rate')
plt.title('Random Forst Model')

# Display legend
plt.legend()

# Show plot
plt.show()


# In[43]:


xx = range(len(y_test))
# Plot actual values
plt.plot(xx, y_test, color='green', label='Actual Prices')

# Plot predicted values
plt.plot(xx, Prediction_rf, color='red', label='Predicted Prices')

# Add labels and title
plt.xlabel('Testing Set Samples')
plt.ylabel('Price')
plt.title('Random Forest Model')
plt.legend()

# Show the plot
plt.show()


# In[44]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
 
# Assuming regressor is your trained Random Forest model
# Pick one tree from the forest, e.g., the first tree (index 0)
tree_to_plot = random_forest_model.estimators_[1]
 
# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree_to_plot, feature_names=df1.columns.tolist(), filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest")
plt.show()


# # AdaBoost

# In[76]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


# Initialize AdaBoost regressor with decision tree as base estimator
base_estimator = DecisionTreeRegressor(max_depth=1)  # Weak learner
adaboost_model = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=50, random_state=42)

# Fit the AdaBoost model
adaboost_model.fit(x_train, y_train)


# In[77]:


mse = mean_squared_error(y_test, adaboost_model.predict(x_test))  # Calculate mean squared error between actual and predicted values
print ("Mean Square Error:",mse)

test_score = r2_score(y_test, adaboost_model.predict(x_test)) * 100
train_score = r2_score(y_train, adaboost_model.predict(x_train)) * 100

print("Test_Score:", test_score)
print("Train_Score:", train_score )


# In[78]:


# Now, let's add a new row to the DataFrame
new_row = ['AdaBoost', train_score, test_score]

# Create a DataFrame for the new row
new_row_df = pd.DataFrame([new_row], columns=results_df.columns)

# Append the new row to the existing DataFrame
results_df = pd.concat([results_df, new_row_df], ignore_index=True)


# In[79]:


#For example, if your new data has features 'feature1' and 'feature2', you need to create a DataFrame with those features
x_validation = pd.DataFrame([[12.2023,85,26,0.2,83,5.4,18593,21668,219096,95300]])

# 2. Use the predict() method of your fitted linear regression model
predictions = adaboost_model.predict(x_validation)

# Now 'predictions' contains the predicted values for the new data
print(predictions)


# In[80]:


Prediction_ad = adaboost_model.predict(x_test)
print(Prediction_ad)
print(y_test)


# Plot actual test data
plt.scatter(y_test.index,y_test, label='Actual', color='blue')

# Plot predicted values
plt.scatter(y_test.index,Prediction_ad, label='Predicted', color='red')

# Set plot labels and title

plt.ylabel('Sulphur Rate')
plt.title('Actual vs Predicted Sulphur Rate')

# Display legend
plt.legend()

# Show plot
plt.show()


# In[81]:


xx = range(len(y_test))
# Plot actual values
plt.plot(xx, y_test, color='pink', label='Actual Prices')

# Plot predicted values
plt.plot(xx, Prediction_ad, color='purple', label='Predicted Prices')

# Add labels and title
plt.xlabel('Testing Set Samples')
plt.ylabel('Price')
plt.title('AdaBoost Model')
plt.legend()

# Show the plot
plt.show()


# # XGBoost Regression

# In[82]:


get_ipython().system('pip install xgboost')



# In[83]:


from xgboost import XGBRegressor


# Initialize XGBoost regressor
xgb_model = XGBRegressor(learning_rate=0.2)

# Fit the XGBoost model
xgb_model.fit(x_train, y_train)


# In[84]:


test_score = r2_score(y_test, xgb_model.predict(x_test)) * 100
train_score = r2_score(y_train, xgb_model.predict(x_train)) * 100

print("Test_Score:", test_score)
print("Train_Score:", train_score )


# In[85]:


# Now, let's add a new row to the DataFrame
new_row = ['XGBoost', train_score, test_score]

# Create a DataFrame for the new row
new_row_df = pd.DataFrame([new_row], columns=results_df.columns)

# Append the new row to the existing DataFrame
results_df = pd.concat([results_df, new_row_df], ignore_index=True)


# In[86]:


#For example, if your new data has features 'feature1' and 'feature2', you need to create a DataFrame with those features

x_validation.columns = ['Months', 'Market Crude Price ($ per Barrel)', 'Avg Temp (Deg C)', 'Precip(cm)', 'Exchange rate', 'Purchase Rate(Sulphuric Acid)', 'Crude Quantity Imported(1000 MT)', 'Total Crude processed (1000 MT)', 'Elemental Sulphur Import Quantity in 1000s(KG)s(Market)', 'Elemental Sulphur Export Quantity in 1000s(KG)s(Market)']
x_validation = pd.DataFrame([[12.2023,85,26,0.2,83,5.4,18593,21668,219096,95300]])

# 2. Use the predict() method of your fitted linear regression model
predictions = xgb_model.predict(x_validation, validate_features=False)

# Now 'predictions' contains the predicted values for the new data
print(predictions)


# In[87]:


Prediction_xg = xgb_model.predict(x_test)
print(Prediction_xg)
print(y_test)


# Plot actual test data
plt.scatter(y_test.index,y_test, label='Actual', color='blue')

# Plot predicted values
plt.scatter(y_test.index,Prediction_xg, label='Predicted', color='red')

# Set plot labels and title

plt.ylabel('Sulphur Rate')
plt.title('XGboost Model')

# Display legend
plt.legend()

# Show plot
plt.show()


# In[88]:


xx = range(len(y_test))
# Plot actual values
plt.plot(xx, y_test, color='blue', label='Actual Prices')

# Plot predicted values
plt.plot(xx, Prediction_xg, color='orange', label='Predicted Prices')

# Add labels and title
plt.xlabel('Testing Set Samples')
plt.ylabel('Price')
plt.title('XGBoost Model')
plt.legend()

# Show the plot
plt.show()


# 

# In[89]:


# Display the updated DataFrame
print(results_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




