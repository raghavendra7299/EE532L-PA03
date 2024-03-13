# EE532L - Deep Learning for Healthcare - Programming Assignment 03
# Authors: Jibitesh Saha, Sasidhar Alavala, Subrahmanyam Gorthi
# Important: Please do not change/rename the existing function names and write your code only in the place where you are asked to do it.


########################################################## Can be modified ##############################################################
# You can import libraries as per your need
import pandas as pd
import sys
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib


# Write your code below so that it returns y_test_pred
def regress_fit(X_train, y_train, X_test, X_val, y_val):
    
    
    



    return y_pred
###########################################################################################################################################


########################################################## Cannot be modified ##############################################################
# Load the dataset
def load_and_fit():
    df = pd.read_csv("data/diabetes.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    # print(df.shape)
    X = df.drop("Outcome", axis=1)
    X2 = np.array(X)
    X2 = X2.T

    y = df["Outcome"]
    X_train = X2[:,:614]
    X_val = X2[:,614:691]
    X_test = X2[:,691:]
    y_train = y[:614].values
    y_val = y[614:691].values
    y_test = y[691:].values

    # Fit the model
    y_test_pred = regress_fit(X_train, y_train, X_test, X_val, y_val)

    # Evaluate the accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.5f}")
    return round(test_accuracy, 5)

a = load_and_fit()
