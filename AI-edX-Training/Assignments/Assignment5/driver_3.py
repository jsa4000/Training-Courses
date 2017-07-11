import os
import sys
import time
import copy
import math
import numpy as np
import pandas as pd
from collections import OrderedDict


if __name__ == "__main__":
    # Define the output file
    output_file = 'output.txt'
    # Boards to be computed
    inputs = []
    # Get the parameters from Args
    if len(sys.argv)<2:
        # If not parameter the define a default string
        with open('sudokus_start.txt','r') as file:
            for line in file:
                inputs.append(line.replace("\n",""))
    else:
        # Only use the current algorithm
        inputs.append(sys.argv[1])

   
      
    # Write current outputs into the output file
    with open(output_file,"w") as file:
        for output in outputs:
            file.write("{}".format(' '.join(output)))

    # End of the Program


