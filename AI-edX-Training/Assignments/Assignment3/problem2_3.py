import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plots import *

def train(data, labels, n, lr=0.05):
    result = []

    return result

if __name__ == "__main__":
    # Start the Program 
    test_mode = True

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


