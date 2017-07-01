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
# Import Models from sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Cross validation with stratified sampling by default
from sklearn.model_selection import cross_val_score

def svm_linear(data,targets):
    '''  SVM with Linear Kernel. 

    Observe the performance of the SVM with linear kernel. Search for a 
    good setting of parameters to obtain high classification accuracy. 
    Specifically.
    
    try values of:
        
        C = [0.1, 0.5, 1, 5, 10, 50, 100]. 

    Read about sklearn.grid_search and how this can help you accomplish 
    this task. After locating the optimal parameter value by using the 
    training data, record the corresponding best score (training data 
    accuracy) achieved. 
    
    Then apply the testing data to the model, and 
    record the actual test score. Both scores will be a number between 
    zero and one.
    '''

    best_score = 0.0
    test_score = 0.0

    return best_score, test_score

def svm_polynomial(data,targets):
    '''SVM with Polynomial Kernel. (Similar to above).
        
    Try values of :

        C = [0.1, 1, 3]
        degree = [4, 5, 6] 
        gamma = [0.1, 0.5]

    '''
    best_score = 0.0
    test_score = 0.0

    return best_score, test_score

def svm_rbf(data,targets):
    ''' SVM with RBF Kernel. (Similar to above).

    Try values of:

        C = [0.1, 0.5, 1, 5, 10, 50, 100] 
        gamma = [0.1, 0.5, 1, 3, 6, 10]

    '''
    best_score = 0.0
    test_score = 0.0

    return best_score, test_score

def logistic(data,targets):
    ''' Logistic Regression. (Similar to above).

    Try values of:
    
        C = [0.1, 0.5, 1, 5, 10, 50, 100]

    '''
    best_score = 0.0
    test_score = 0.0

    return best_score, test_score

def knn(data,targets):
    ''' k-Nearest Neighbors. (Similar to above).

    Try values of:

        n_neighbors = [1, 2, 3, ..., 50] 
        leaf_size = [5, 10, 15, ..., 60]

    '''
    best_score = 0.0
    test_score = 0.0

    return best_score, test_score

def decision_tree(data,targets):
    ''' Decision Trees. (Similar to above).

    Try values of:

        max_depth = [1, 2, 3, ..., 50]
        min_samples_split = [2, 3, 4, ..., 10]

    '''
    best_score = 0.0
    test_score = 0.0

    return best_score, test_score

def random_forest(data,targets):
    ''' Random Forest. (Similar to above).

    Try values of:
    
         max_depth = [1, 2, 3, ..., 50] 
         min_samples_split = [2, 3, 4, ..., 10]

    '''
    best_score = 0.0
    test_score = 0.0

    return best_score, test_score



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
    data = df.loc[:,["A","B"]]
    targets = df.loc[:,"label"]
   
    # Create models to use cross-validation
    models = [svm_linear,svm_polynomial,svm_rbf,logistic,knn,decision_tree,random_forest]
    scores =  []
    for model in models:
        scores.append(model(data,targets))

    # Write current outputs into the file
  # Write current weights into the file
    with open(output_file,"w") as file:
        for index, score in enumerate(scores):
            file.write("{},{}\n".format(models[index].__name__,",".join(format(score_type, "0.8f") for score_type in score)))

    # End of the Program


