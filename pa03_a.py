# EE532L - Deep Learning for Healthcare - Programming Assignment 03
# Authors: Jibitesh Saha, Sasidhar Alavala, Subrahmanyam Gorthi
# Important: Please do not change/rename the existing function names and write your code only in the place where you are asked to do it.

########################################################## Can be modified ##############################################################
# You can import libraries as per your need
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


# Write your code below so that it returns y_test_pred
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def regress_fit(X_train, y_train, X_test, X_val, y_val):
    
##    X_train = np.array(X_train) # Normalizing training features
##    c = np.expand_dims(np.amax(X_train, axis=1),axis=1)
##    X_train = X_train / c
##
##    X_val = np.array(X_val) # Normalizing validating features
##    c = np.expand_dims(np.amax(X_val, axis=1),axis=1)
##    X_val = X_val / c
##    
##    X_test = np.array(X_test) # Normalizing testing features
##    c = np.expand_dims(np.amax(X_test, axis=1),axis=1)
##    X_test = X_test / c
    # Normalizing training features
    X_train = (X_train - np.mean(X_train, axis=1, keepdims=True)) / np.std(X_train, axis=1, keepdims=True)

    # Normalizing validating features
    X_val = (X_val - np.mean(X_val, axis=1, keepdims=True)) / np.std(X_val, axis=1, keepdims=True)

    # Normalizing testing features
    X_test = (X_test - np.mean(X_test, axis=1, keepdims=True)) / np.std(X_test, axis=1, keepdims=True)


    p = X_train.shape[0] # Number of features
    N1 = X_train.shape[1] # Number of train cases
    
    N2 = X_val.shape[1] # Number of val cases
    # Initialize weights and biases
    np.random.seed(42)
    input_size = X_train.shape[0]
    hidden_size = 3
    output_size = 1
    learning_rate = 0.1
    epochs = 20000

    losses = []
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    ious = []

    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_specificities = []
    val_f1_scores = []
    val_ious = []

    TP,TN,FP,FN = 0,0,0,0

    W1 = np.random.randn(hidden_size, input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size)
    b2 = np.zeros((output_size, 1))

    train_losses = []

    for epoch in range(epochs):
        J = 0 # Initializing loss
        count = 0 # Initializing count of correct predictions
        # Forward pass
        Z1 = np.dot(W1, X_train) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)

        # Compute loss
        train_loss = -np.mean(y_train * np.log(A2) + (1 - y_train) * np.log(1 - A2 ))
        losses.append(train_loss)

        # Backward pass
        dZ2 = A2 - y_train
        dW2 = np.dot(dZ2, A1.T) / X_train.shape[1]
        db2 = np.sum(dZ2, axis=1, keepdims=True) / X_train.shape[1]
        dA1 = np.dot(W2.T, dZ2)
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = np.dot(dZ1, X_train.T) / X_train.shape[1]
        db1 = np.sum(dZ1, axis=1, keepdims=True) / X_train.shape[1]

        # Update weights and biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

        diff = np.abs(A2 - y_train)
        count = np.sum(diff < 0.5)
                        
            
        TP = np.sum((A2 >= 0.5) & (y_train == 1))
        TN = np.sum((A2 < 0.5) & (y_train == 0))
        FP = np.sum((A2 >= 0.5) & (y_train == 0))
        FN = np.sum((A2 < 0.5) & (y_train == 1))

        train_accuracy = 100*count/N1
        
        accuracies.append(train_accuracy)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        precisions.append(precision)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        recalls.append(recall)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificities.append(specificity)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        ious.append(iou)

        # Validation
        Z1_val = np.dot(W1, X_val) + b1
        A1_val = sigmoid(Z1_val)
        Z2_val = np.dot(W2, A1_val) + b2
        A2_val = sigmoid(Z2_val)
        

        diff = np.abs(A2_val - y_val)
        count = np.sum(diff < 0.5)
                        
            
        TP = np.sum((A2_val >= 0.5) & (y_val == 1))
        TN = np.sum((A2_val < 0.5) & (y_val == 0))
        FP = np.sum((A2_val >= 0.5) & (y_val == 0))
        FN = np.sum((A2_val < 0.5) & (y_val == 1))
          
        
        accuracy = 100*count/N2
        val_accuracies.append(accuracy)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        val_precisions.append(precision)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        val_recalls.append(recall)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        val_specificities.append(specificity)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        val_f1_scores.append(f1_score)
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        val_ious.append(iou)

    #batch_metrics = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} \n"
    #sys.stdout.write('\r' + batch_metrics)
    #sys.stdout.flush()

  
    epochs = np.arange(1, epochs+1)
    
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')

    plt.subplot(2, 3, 2)
    plt.plot(epochs, accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')

    plt.subplot(2, 3, 3)
    plt.plot(epochs, precisions)
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision vs Epoch')

    plt.subplot(2, 3, 4)
    plt.plot(epochs, recalls)
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall vs Epoch')

    plt.subplot(2, 3, 5)
    plt.plot(epochs, specificities)
    plt.xlabel('Epoch')
    plt.ylabel('Specificity')
    plt.title('Specificity vs Epoch')

    plt.subplot(2, 3, 6)
    plt.plot(epochs, f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epoch')

    plt.tight_layout()
    plt.show()


    #batch_metrics = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} \n"
    #sys.stdout.write('\r' + batch_metrics)
    #sys.stdout.flush()

  
    #epochs = np.arange(1, epochs+1)
    
    plt.figure(figsize=(12, 5))



    plt.subplot(2, 3, 1)
    plt.plot(epochs, val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Val_Accuracy vs Epoch')

    plt.subplot(2, 3, 2)
    plt.plot(epochs, val_precisions)
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Val_Precision vs Epoch')

    plt.subplot(2, 3, 3)
    plt.plot(epochs, val_recalls)
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Val_Recall vs Epoch')

    plt.subplot(2, 3, 4)
    plt.plot(epochs, val_specificities)
    plt.xlabel('Epoch')
    plt.ylabel('Specificity')
    plt.title('Val_Specificity vs Epoch')

    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Val_F1 Score vs Epoch')

    plt.tight_layout()
    plt.show()


    # Predictions on test set
    Z1_test = np.dot(W1, X_test) + b1
    A1_test = sigmoid(Z1_test)
    Z2_test = np.dot(W2, A1_test) + b2
    A2_test = sigmoid(Z2_test)
    y_test_pred = (A2_test > 0.5).astype(int).flatten()
    #y_test_pred=A2_test
    #y_test_pred[y_test_pred>=0.5] = 1 # Thresholding
    #y_test_pred[y_test_pred<0.5] = 0

    return y_test_pred
###########################################################################################################################################

# Load the dataset
def load_and_fit():
    df = pd.read_csv("diabetes.csv")
    df = df.sample(frac=1).reset_index(drop=True)
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
