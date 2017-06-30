import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plots import *


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
    df = pd.read_csv(input_file)
    if test_mode: test_dataset(df) # If test mode enabled (2D)

    # Get training and labels from dataset
    training_set = df.loc[:,["A","B"]]
    labels = df.loc[:,"label"]
   

    # End of the Program


