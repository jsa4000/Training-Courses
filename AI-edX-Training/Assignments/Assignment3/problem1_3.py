import os
import sys
import time
import math
import pandas as pd
import matplotlib 


if __name__ == "__main__":
    # Start the Programs    

    # Get the parameters from the Args
    if len(sys.argv)<3:
        print("Not enought paramters. e.g file.py input.csv output.csv")
        sys.exit()
    # Get the Parameters and prepare the input/output files
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    #print("input: {} | output: {}".format(input_file,output_file))

    # Read the input file and extract the features
    df = pd.read_csv(input_file, names=["x","y","label"],header=None)
    print("Number of Samples in the Dataset: {}".format(len(df.index)))
    print("Number of Features in the Dataset: {}".format(len(df.columns)))
    print(df.head(1)) # First Element read

