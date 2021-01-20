# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 18:41:57 2021

@author: ceosh
"""


import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.api as sm
import numpy as np
st.title('Model Deployment: Housing Price')

st.sidebar.header('User Input Parameters')

def user_input_features():
    area = st.sidebar.number_input("Insert the area")
    bedrooms = st.sidebar.number_input("Insert the number of bedrooms")
    bathrooms = st.sidebar.number_input("Insert the number of bathrooms")                       
    stories = st.sidebar.number_input("Insert the number of stories")
    mainroad = st.sidebar.selectbox('mainroad',('yes','no'))
    guestroom = st.sidebar.selectbox('guestroom',('yes','no'))
    basement = st.sidebar.selectbox('basement',('yes','no'))
    hotwaterheating = st.sidebar.selectbox('hotwaterheating',('yes','no'))
    airconditioning = st.sidebar.selectbox('airconditioning',('yes','no'))
    parking = st.sidebar.number_input("Insert the number of parking")
    prefarea = st.sidebar.selectbox('prefarea',('yes','no'))
    semi_furnished = st.sidebar.selectbox('semi-furnished',('yes','no'))
    unfurnished = st.sidebar.selectbox('unfurnished',('yes','no'))
    
    data = {'area':area,
            'bedrooms':bedrooms,
            'bathrooms':bathrooms,
            'stories':stories,
            'mainroad':mainroad,
            'guestroom':guestroom,
            'basement':basement,
            'hotwaterheating':hotwaterheating,
            'airconditioning':airconditioning,
            'parking':parking,
            'prefarea':prefarea,
            'semi-furnished':semi_furnished,
            'unfurnished' : unfurnished}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

#Read the data
housing = pd.read_csv("G:\\Jupyter\\jupyter\\Deployment\\Demo_Shivam\\Housing.csv")

varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the housing list
housing[varlist] = housing[varlist].apply(binary_map)

# Get the dummy variables for the feature 'furnishingstatus' and store it in a new variable - 'status'
status = pd.get_dummies(housing['furnishingstatus'])

# Let's drop the first column from status df using 'drop_first = True'

status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)

# Add the results to the original housing dataframe

housing = pd.concat([housing, status], axis = 1)

# Drop 'furnishingstatus' as we have created the dummies for it

housing.drop(['furnishingstatus'], axis = 1, inplace = True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

housing[num_vars] = scaler.fit_transform(housing[num_vars])

Y=housing.iloc[:,0] # Target Variable
X=housing.iloc[:,1:14]

from sklearn.model_selection import train_test_split
np.random.seed(0)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=100)
#Build model
import statsmodels.api as sm
# Add a constant
X_train_lm = sm.add_constant(X_train)
model1 = sm.OLS(y_train, X_train_lm).fit()
print(model1.summary())



##Testing ####

# Applying the function to the housing list
varlist1 =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'semi-furnished','unfurnished']
df[varlist1] = df[varlist1].apply(binary_map)


# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars1 = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

df1=sm.add_constant(df.iloc[0:1,], has_constant='add')

### Predicting ###
y = model1.predict(df1)


st.subheader('Predicted Result')
st.write(y)

























