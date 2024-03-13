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
    
    p = X_train.shape[0] # Number of features
    q =  # Number of neurons in hidden layer
    N = X_train.shape[1] # Number of sample cases
    e =  # Learning Rate (Hint:- adjust between 1e-5 to 1e-2)
    
    w1 = np.random.uniform(-1/np.sqrt(p), 1/np.sqrt(p), (q,p+1)) # Random initialization of weights
    w2 = np.random.uniform(-1/np.sqrt(q), 1/np.sqrt(q), (1,q+1))
    X = np.ones((p+1,N)) # Adding an extra column of ones to adjust biases on training features
    X[:p,:] = X_train
    
    N3 = X_val.shape[1] # Number of validation samples

    X3 = np.ones((p+1,N3)) # adding an additional columns of ones to adjust biases on validation features
    X3[:p,:] = X_val
   
    num_epochs = 
    for epoch in range(num_epochs): # Loop for iterative process

        # TRAINING
        J = 0 # Initializing loss
        count = 0 # Initializing count of correct predictions
        for i in range (N):






        # VALIDATION       
        J2 = 0
        count2 = 0        
                
        for i in range (N3):



        
        
        
        
        train_loss = J/N
        train_accuracy = 100*count/N
        val_loss = J2/N3
        val_accuracy = 100*count2/N3
        
        # print(epoch,J/N,100*count/N)
        batch_metrics = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}"
        sys.stdout.write('\r' + batch_metrics)
        sys.stdout.flush()
    
    # TESTING
    print("\n")
    N2 = X_test.shape[1] # Number of test samples

    X2 = np.ones((p+1,N2)) # adding an additional columns of 1 to adjust biases
    X2[:p,:] = X_test
    z2 = np.zeros(N2)
    for i in range (N2):
        



    y_pred = 1*(z2>=0.5)
    



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
