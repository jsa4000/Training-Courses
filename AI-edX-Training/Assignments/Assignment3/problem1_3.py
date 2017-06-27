import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_points(df, plot_margin = 2):
    """ This Function Plot the x, y points from the dataframe
    Also color changes depending on the label class for each point
    """
    # Set data: point (x, y) and colors for classification
    plt.scatter(df["x"],df["y"],c=df["label"])
    # Set the margrins with some space
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - plot_margin, x1 + plot_margin,
            y0 - plot_margin, y1 + plot_margin))
    # Show the plot in a pop window
    plt.show()

def test_dataset(df):
    print("Number of Samples in the Dataset: {}".format(len(df.index)))
    print("Number of Features in the Dataset: {}".format(len(df.columns)))
    print(df.head(1)) # First Element read
    # Plot dataset Points
    plot_points(df)


def compute_PLA(df):
    """ Perceptron Learning Algorithm

      ... X2  X1   W1                                      
                     PERCEPTRON + LOGISTIC FUNCTION => PREDICTION
      ... Y2  Y1   W2    

        y = f(z), where z is the input vector (X1,Y2)

        f = sigmoid(X1*W1 + X2*W2 + B)

        Training Set: [(X1,Y1), (X2,Y2), .. , (XN, YN)]
        Labels:       [Y1, Y2, ..., YN]


    """
    # Initialize weights (zeros, ones, random, uniform...)
    weights = np.ones((2,1))
    # Get training and labels from dataset
    training_set = df.loc[:,["x","y"]]
    labels = df.loc[:,"label"]
    # Matrix Multiplication (N,2)) x (2,1) -> N:1
    mult = np.matmul(training_set, weights)
  
    return True



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
    # Compute PLA
    compute_PLA(df)
    # End of the Program



