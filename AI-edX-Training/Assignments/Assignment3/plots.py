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
  
def plot_function(vector, y, line = None):
    if line is not None: line.remove()
    x = np.arange(vector[0], vector[1], 1.0)
    y = list(map(y,x))
    line, = plt.plot(x, y, lw=2)
    return line;

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
