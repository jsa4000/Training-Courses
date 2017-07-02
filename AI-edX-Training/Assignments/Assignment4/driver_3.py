import os
import sys
import time
import numpy as np
import pandas as pd


if __name__ == "__main__":
    # Start the Program 
    test_mode = False

    # Get the parameters from the Args
    if len(sys.argv)<2:
        print("Not enought paramters. e.g file.py input_file.txt")
        sys.exit()

    # Get the Parameters and prepare the input/output files
    input_file = sys.argv[1]
    output_file = "output.txt"
  
    if test_mode: print("in: {}".format(input_file))

    # Read the inputs and compute the game
    results = []
    with open(input_file,"r") as file:
        for line in file:s
            results.append(line)
    
    # Write current outputs into the output file
    with open(output_file,"w") as file:
        for index, result in enumerate(results):
            file.write("{},{}".format(index,result))


    # End of the Program


