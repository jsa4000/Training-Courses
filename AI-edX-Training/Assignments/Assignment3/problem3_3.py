import os
import sys
import time
# Standard libaries for data manipulation
import numpy as np
import pandas as pd
# Plot imports
import matplotlib.pyplot as plt
from plots import *
'''
 Import scikit-learn models: http://scikit-learn.org/stable/supervised_learning.html

 - SVM with Linear Kernel. Observe the performance of the SVM with linear kernel. 
 - SVM with Polynomial Kernel. (Similar to above).
 - SVM with RBF Kernel. (Similar to above).

    Use svm.SVC(kernel='%option%')
    Where %option%:
        linear: \langle x, x'\rangle.
        polynomial: (\gamma \langle x, x'\rangle + r)^d. d is specified by keyword degree, r by coef0.
        rbf: \exp(-\gamma |x-x'|^2). \gamma is specified by keyword gamma, must be greater than 0.
        sigmoid (\tanh(\gamma \langle x,x'\rangle + r)), where r is specified by coef0.

 - Logistic Regression. (Similar to above).
 - k-Nearest Neighbors. (Similar to above).
 - Decision Trees. (Similar to above). 
 - Random Forest. (Similar to above).
  
'''
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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


