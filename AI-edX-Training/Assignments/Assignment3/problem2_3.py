import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plots import *

def predict(data, weights):
    # Sum (xi * wi) + Bias => data * weights (bias included)
    # Matrix Multiplication: 
    # (N,3)) x (3,1) -> N:1
    return np.matmul(data, weights) # Linear function y'(t)
 
def update(data, labels, weights, lr=0.05):
    """ Linear Regression

    """
    # Error distance (based on square error gradient descent)
    error = (labels - predict(data, weights)) / len(labels)
    # Compute the Update for the weigths (for each row in order)
    return weights - lr * np.matmul(error, data.values) 

def train(data, labels, n, lr=0.05):
    result = []
    # Initialize weights (zeros, ones, random, uniform...)
    # x, y and Bias (W0, W1 and W2)
     # Create the parameter for the Bias term 1 * W0 = W0
    data["b"] = 1
    # Get training and labels from dataset
    data = data.loc[:,["age","weight","b"]]
    weights = np.zeros(3,) 
    #Start with the TRaining
    line = None
    # Plot the results interactively
    plt.ion()
    plot_points(df)
    plot_margin()
    # Iterate througn n iterations
    for i in range(n):
        # Update PLA and update the weights
        weights = update(data, labels, weights, lr)
        # Plot current weights
        line = plot_function((-20, 20), lambda x:(weights[0]*x + weights[2])/weights[1], line)
        # Update the plot (refresh)
        plt.pause(0.05)
        # Add current weight
        result.append(weights)
        
    # Show the final 10 seconds
    plt.pause(1)
    #Return the collection with all the weights updates
    return result;

if __name__ == "__main__":
    # Start the Program 
    test_mode = False

    # Get the parameters from the Args
    if len(sys.argv)<3:
        print("Not enought paramters. e.g file.py input.csv output.csv")
        sys.exit()

    # Get the Parameters and prepare the input/output files
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    if test_mode: print("in: {} | out: {}".format(input_file,output_file))

    # Read the input file and extract the features
    df = pd.read_csv(input_file, names=["age","weight","height"],header=None)
    if test_mode: test_dataset(df) # If test mode enabled (2D)

    # Normalize the data (each feature): 
    # norm(x) = (x - mean(x)) / std(x)
    df.loc[:,"age"] = (df.loc[:,"age"] - df.loc[:,"age"].mean()) / df.loc[:,"age"].std()
    df.loc[:,"weight"] = (df.loc[:,"weight"] - df.loc[:,"weight"].mean()) / df.loc[:,"weight"].std()
    df.loc[:,"height"] = (df.loc[:,"height"] - df.loc[:,"height"].mean()) / df.loc[:,"height"].std()
    if test_mode: test_dataset(df, True) # If test mode enabled (3D)
    
    # Get training and labels from dataset
    training_set = df.loc[:,["age","weight"]]
    labels = df.loc[:,"height"]

    # Train the data set for n iterations
    weights = train(training_set, labels, 100, 0.01)

    # Write current weights into the file
    with open(output_file,"w") as file:
        for weight in weights:
            file.write(",".join(format(x, "10.3f") for x in weight))
            file.write('\n')

    # End of the Program


