##########################
#   Description: Just a IA that predicts the money you earn when you sells a product that cost is 1
#                Client says the amount of products, IA predicts the price.
#
#   Steps:
#       1) Import modules (numpy: working with numbers, sklearn: machine learning, pandas: data science) (X)
#       2) Make some data (X)
#       3) Make a linear regression model (X)
#       4) Fit the model
#       5) Evaluate the model
#       6) Make avalaible for user
##########################

# 1) Import modules
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Make some data
dataframe = pd.read_csv('data.csv')
valueX = dataframe[['X']]
valueY = dataframe[['Y']]

# Make a linear regression model
model = LinearRegression()

# Fit the model
print('Training...')
model.fit(valueX, valueY)
print('Trained!')

# Evaluate the model
print('Evaluating...')
score = model.score(valueX, valueY)
print(f'The model score is: {str(int(score))}/1')

# Make avalaible for user
products = input('Input the amount of products: ')
print('Thinking...')
price = model.predict(np.array([[float(products)]], dtype=float))
print(f'The price is: {str(int(price))}')