import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_points(df):
    """ This Function Plot the x, y points from the dataframe
    Also color changes depending on the label class for each point
    """
    # Set data: point (x, y) and colors for classification
    plt.scatter(df["x"],df["y"],c=df["label"])
  
def plot_function(y):
    x = np.arange(0.0, 20.0, 1.0)
    y = list(map(y,x))
    line, = plt.plot(x, y, lw=2)

def plot_margin(plot_margin = 2):
    # Set the margrins with some space
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - plot_margin, x1 + plot_margin,
            y0 - plot_margin, y1 + plot_margin))

def test_dataset(df):
    print("Number of Samples in the Dataset: {}".format(len(df.index)))
    print("Number of Features in the Dataset: {}".format(len(df.columns)))
    print(df.head(1)) # First Element read
    # Plot dataset Points
    plot_points(df)
    plot_margin()
    plt.show()

def compute_PLA(df, weights=None, lr=0.05):
    """ Perceptron Learning Algorithm

    PLA is based in Linear Regression. The main purpose
    is to divide the dataset into two. In order to do this
    the data must be linearly separable.

                1    W0 -> Bias
      (... X2, X1) * W1                                      
                      + PERCEPTRON + LINEAR FUNCTION => PREDICTION
      (... Y2, Y1) * W2   

        y = f(z), where z is the input vector (X1,Y2)

        f =  linear (X1*W1 + X2*W2 + W0)

        Training Set: [(X1,Y1), (X2,Y2), .. , (XN, YN)]
        Labels:       [Y1, Y2, ..., YN]
        Bias = Weight * 1 (W0)

    In order to update the weights:

    W(t+1) = Wt + lr(y - y'(t))xi
    =>
    AW =  lr (y - y'(t))xi
    =>
    W(t+1) = Wt + AWt

    where:
        lr: learning rate
        y: true label
        y'(t): predicted label (t)
        xi: parameter to the weight being updated.
        Wt: previous weight (t)
        W(t+1): update weights

    """
    # Initialize weights (zeros, ones, random, uniform...)
    # x, y and Bias (W0, W1 and W2)
    if weights is None: weights = np.random.rand(3,)  
    # Create the parameter for the Bias term 1 * W0 = W0
    df["b"] = 1
    # Get training and labels from dataset
    training_set = df.loc[:,["x","y","b"]]
    labels = df.loc[:,"label"]
    # Matrix Multiplication (N,3)) x (3,1) -> N:1
    mult = np.matmul(training_set, weights) # Linear function y'(t)
    # Error distance (based on square error gradient descent)
    error = labels - mult
    # Compute the Update for the weigths (for each row in order)
    for i, label in enumerate(labels):
        weights = weights + lr * error[i] * training_set.loc[i,:] 
    return weights.values

if __name__ == "__main__":
    # Start the Programs    
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
    df = pd.read_csv(input_file, names=["x","y","label"],header=None)
    if test_mode: test_dataset(df) # If test mode enabled
    
    weights = None

    for i in range(100):
        # Compute PLA
        weights = compute_PLA(df, weights)

        # Plot the results
        plt.ion()
        plot_points(df)
        plot_function(lambda x: weights[0]*x**2 + weights[1]*x + weights[2] )
        plot_margin()
        plt.pause(0.05)

    # End of the Program


