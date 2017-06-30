import os
import sys
import numpy as np
import pandas as pd

def predict(data, weights):
    # Sum (xi * wi) + Bias => data * weights (bias included)
    # Matrix Multiplication: 
    # (N,3)) x (3,1) -> N:1
    mult = np.matmul(data, weights) # Linear function y'(t)
    activation = lambda x: 1.0 if x>=0 else -1.0
    nv = np.vectorize(activation)
    return nv(mult)

def update(data, labels, weights, lr=0.05):
    """ Perceptron Learning Algorithm

    PLA is based in Linear Regression. The main purpose
    is to divide the dataset into two. In order to do this
    the data must be linearly separable.

                1    W0 -> Bias
      (... X2, X1) * W1                                      
                      + PERCEPTRON + LINEAR FUNCTION => PREDICTION
      (... Y2, Y1) * W2   

        ACTIVATION:  if x >= 1   1
                     else        0

        
        ACTIVATION (2):

        In a Perceptron (PLA) the formule is the following. (W3 = Bias)
        Where X1, X2 are the features and Y are the labels

            X1W1 + X2W2 + W3 = 0 

            In PLA the idea is to resolve the activateion (sign) function.
            Previous formule since it's equal to 0, this means the 
            line that cross (divide) between the cloud of points.

            if X1W1 + X2W2 + W3 >= 1    label = 1
            if X1W1 + X2W2 + W3 < 1    label = 0

        In a Linear Regreseion the basic formule will be:

             X1W1 + X2W2 + X3W3 + W4 = 0

             This is the eigenvector that will fit over the cloud of
             points gven the features (X1,X2 abd X3). In this case the
             label will also have a weight that will be computed.

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
    # Error distance (based on square error gradient descent)
    error = labels - predict(data, weights)
    # Compute the Update for the weigths (for each row in order)
    for i, label in enumerate(labels):
        weights = weights + lr * error[i] * data.loc[i,:] 
    return weights

def train(data, labels, n, lr=0.05):
    result = []
    # Initialize weights (zeros, ones, random, uniform...)
    # x, y and Bias (W0, W1 and W2)
     # Create the parameter for the Bias term 1 * W0 = W0
    data["b"] = 1
    # Get training and labels from dataset
    data = df.loc[:,["x","y","b"]]
    weights = np.random.rand(3,) 
    #weights = np.zeros(3,) 
    #Start with the TRaining
    # Iterate througn n iterations
    for i in range(n):
        # Update PLA and update the weights
        weights = update(data, labels, weights, lr)
        # Add current weight
        result.append(weights)
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

    # Read the input file and extract the features
    df = pd.read_csv(input_file, names=["x","y","label"],header=None)

    # Get training and labels from dataset
    training_set = df.loc[:,["x","y"]]
    labels = df.loc[:,"label"]

    # Train the data set for n iterations
    weights = train(training_set, labels, 50, 0.01)

    # Write current weights into the file
    with open(output_file,"w") as file:
        for weight in weights:
            file.write(",".join(format(x, "10.3f") for x in weight))
            file.write('\n')

    # End of the Program


