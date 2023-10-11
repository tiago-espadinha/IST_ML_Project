'''
Machine Learning - Image Analysis Problem 1
Authors:
Tiago Simões, 96329
Tomás Fonseca, 66325
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_validate

# Change path to the folder where the data is located
path = 'Class1/'

# Use StandardScaler to scale the data
scale = True

def main():

    # Load datasets
    X_train = np.load(path + 'Xtest_Classification1.npy')
    Y_train = np.load(path + 'ytrain_Classification1.npy')
    X_test = np.load(path + 'Xtest_Classification1.npy')

    X_scaler = StandardScaler()
    Y_scaler = StandardScaler() 

    if scale:
        X_train = X_scaler.fit(X_train).transform(X_train)
        Y_train = Y_scaler.fit(Y_train).transform(Y_train)
        X_test = X_scaler.transform(X_test)


if __name__ == "__main__":
    main()