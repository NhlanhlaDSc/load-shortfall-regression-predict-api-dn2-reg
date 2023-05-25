"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""
'''
# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed','Bilbao_wind_speed']]

# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/load_shortfall_simple_lm_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))'''

############## our base model starts here ###############

import numpy as np
import pandas as pd 
import math 
import pickle
# libraries for modelling and model evaluation
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
import statsmodels.formula.api as sm 
from statsmodels.formula.api import ols
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor



df_train = pd.read_csv("./data/df_train.csv")
df_test = pd.read_csv("./data/df_test.csv")
print(df_test.head())
#print(f"The train set has {df_train.shape[0]} rows and {df_train.shape[1]} columns")
#print(f"The test set has {df_test.shape[0]} and {df_test.shape[1]} columns")

combined = pd.concat([df_train, df_test])
#print(f"The combined dataset has {combined.shape[0]} rows and {combined.shape[1]} columns")


combined['Valencia_pressure'] = combined['Valencia_pressure'].fillna(combined['Valencia_pressure'].mode()[0])
combined['load_shortfall_3h'] = combined['load_shortfall_3h'].fillna(combined['load_shortfall_3h'].mode()[0])


combined.dtypes #to retrieve the data types of each column in the combined dataframe.
combined['time']


combined['time'] = pd.to_datetime(combined['time'])

combined.time


combined['year'] = combined['time'].dt.year
combined['month'] = combined['time'].dt.month
combined['day'] = combined['time'].dt.day
combined['hour'] = combined['time'].dt.hour
combined['day_of_week'] = combined['time'].dt.dayofweek
combined['time_diff_minutes'] = combined['time'].diff().dt.total_seconds() / 60.0
combined['hour_sin'] = np.sin(2 * np.pi * combined['hour'] / 24)
combined['hour_cos'] = np.cos(2 * np.pi * combined['hour'] / 24)
combined['Valencia_wind_deg']
combined['Valencia_wind_deg'] = combined['Valencia_wind_deg'].astype(str)
combined['Valencia_wind_deg'] = combined['Valencia_wind_deg'].str.extract(r'(\d+)')
combined['Valencia_wind_deg']
combined['Valencia_wind_deg'] = pd.to_numeric(combined['Valencia_wind_deg'])
combined['Valencia_wind_deg']
combined.Seville_pressure


combined['Seville_pressure'] = combined['Seville_pressure'].astype(str)

# Extract numeric values from the 'Seville_pressure' column

combined['Seville_pressure'] = combined['Seville_pressure'].str.extract(r'(\d+)')

# Convert 'Seville_pressure' column to integer type

combined['Seville_pressure'] = pd.to_numeric(combined['Seville_pressure'])

combined.Seville_pressure
combined = combined.drop(['Unnamed: 0' , 'time'], axis = 1) #time data is very important, refer to @ time 32 min of the video 

combined =combined[['Madrid_wind_speed', 'Valencia_wind_deg', 'Bilbao_rain_1h', 'Valencia_wind_speed','load_shortfall_3h']]
#training the model
y = combined[:len(df_train)][['load_shortfall_3h']]
x = combined[:len(df_train)].drop('load_shortfall_3h',axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,random_state =2)
#x_test["time_diff_minutes"] = x_test["time_diff_minutes"].fillna(x_test["time_diff_minutes"].mean())
#x_train["time_diff_minutes"] = x_train["time_diff_minutes"].fillna(x_train["time_diff_minutes"].mean())

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  #creating the Random forest model
rf_model.fit(x_train, y_train)    # #fitting training data to the model
rf_model_predictions = rf_model.predict(x_test)   #generating predictions from the model
#reg_model = LinearRegression() #creating the Linear regression model
#reg_model.fit(x_train, y_train) #fitting training data to the model
#reg_model_predictions = reg_model.predict(x_test) #generating predictions from the model
#model evaluation
from sklearn.metrics import mean_squared_error
def calculate_rmse(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


calculate_rmse(y_test,rf_model_predictions)
r2_score(y_test, rf_model_predictions)


x_train = combined[:len(df_train)].drop('load_shortfall_3h',axis = 1)
x_test = combined[len(df_train):].drop('load_shortfall_3h',axis = 1)
#x_train["time_diff_minutes"] = x_train["time_diff_minutes"].fillna(x_train["time_diff_minutes"].mean())

#fitting_the_model
rf_model.fit(x_train,y)
#pickle file
save_path = '../assets/trained-models/load_shortfall_base_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(rf_model, open(save_path,'wb'))