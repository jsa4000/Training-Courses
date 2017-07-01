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

# Train-test split and Cross validation with stratified sampling functions
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

def fit_model_train_test(classifier, data, targets, test_size = 0.4):
    ''' Train/Test Split

    The datais are split into training data and test data. 
    
        - The training set contains a known output and the model learns on this 
        data in order to be generalized to other data later on.
        
        - The test dataset (or subset) in order to test our model’s prediction 
        on this subset.

    '''
    # Split data into train and tests (by (test_size * 100) percentage)
    X_train, X_test, Y_train, Y_test = train_test_split(data, targets, test_size=test_size)
    # Fit the model using the training data splitted previously by using the classifier
    model = classifier.fit(X_train, Y_train)
    # Get the predcition with the trained classifier and ussing the test data
    predictions = classifier.predict(X_test)
    #REturn the score (accuracy) of the model trainnied using the test data set
    return model.score(predictions, Y_test)    

def fit_model_cross_validation(classifier, data, targets, kfold=10):
    ''' Cross Validation
    
    It’s very similar to train/test split, but it’s applied to more subsets. Meaning, 
    we split our data into k subsets, and train on k-1 one of those subset. What we 
    do is to hold the last subset for test. We’re able to do it for each of the subsets.

    There are a bunch of cross validation methods: K-Folds Cross Validation, Leave One Out 
    Cross Validation (LOOCV), etc..

    K-Folds Cross Validation. In this method we split our data into k different subsets 
    (or folds). We use k-1 subsets to train our data and leave the last subset (or the 
    last fold) as test data. We then average the model against each of the folds and then
    finalize our model. After that we test it against the test set.

    Also, we will be using Stratified sampling, that is a probability sampling technique 
    wherein the researcher divides the entire population into different subgroups or strata,
    then randomly selects the final subjects proportionally from the different strata. THe data
    and target and proportionally divided (balanced) using this method.
    
    '''
    # Evaluate a score by cross-validation (Stratified sampling)
    scores = cross_val_score(classifier, data, targets, parameters, cv=kfold)
    # Return the best score obtained in the Cross validation
    return np.max(scores)

def fit_model_grid_search(classifier, data, targets, parameters, kfold=10, test_size=0.4):
    ''' GridSearchCV
    Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a “fit” and a “score” method. It also implements 
    “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform”
    if they are implemented in the estimator used.

    The parameters of the estimator used to apply these methods are optimized by 
    cross-validated grid-search over a parameter grid.

    '''
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(data, targets, 
                                    test_size=test_size, random_state=0)
    # Use GridSearchCV for Exhaustive search over specified parameter values for an estimator
    model = GridSearchCV(classifier, parameters, cv=kfold)
    # Train the model with the train data
    model.fit(X_train, y_train)
    # Get the scores, best, mean_test, standard deviation, etc..
    test_cores = model.cv_results_['mean_test_score']
    # Predcit using the file model and the training test
    y_true, y_pred = y_test, model.predict(X_test)
    #Return the best train and test mean scores from cross validation
    return accuracy_score(y_true, y_pred), np.max(test_cores)

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
    # Set the parameters by cross-validation
    parameters = [ {'kernel': ['linear'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100]}]
    # Create linea regression classifier
    classifier = SVC()
    # Fit the model using grid search and get the best train and test scores
    return fit_model_grid_search(classifier, data, targets, parameters, 
                                kfold=5, test_size=0.4)

def svm_polynomial(data,targets):
    '''SVM with Polynomial Kernel. (Similar to above).
        
    Try values of :

        C = [0.1, 1, 3]
        degree = [4, 5, 6] 
        gamma = [0.1, 0.5]

    '''
    # Set the parameters by cross-validation
    parameters = [ {'kernel': ['poly'], 'gamma':[0.1, 0.5], 
                    'C':[0.1, 1, 3], 'degree': [4, 5, 6] }]
    # Create linea regression classifier
    classifier = SVC()
    # Fit the model using grid search and get the best train and test scores
    return fit_model_grid_search(classifier, data, targets, parameters, 
                                kfold=5, test_size=0.4)

def svm_rbf(data,targets):
    ''' SVM with RBF Kernel. (Similar to above).

    Try values of:

        C = [0.1, 0.5, 1, 5, 10, 50, 100] 
        gamma = [0.1, 0.5, 1, 3, 6, 10]

    '''
    # Set the parameters by cross-validation
    parameters = [ {'kernel': ['rbf'], 'gamma':  [0.1, 0.5, 1, 3, 6, 10], 
                    'C': [0.1, 0.5, 1, 5, 10, 50, 100]}]
    # Create linea regression classifier
    classifier = SVC()
    # Fit the model using grid search and get the best train and test scores
    return fit_model_grid_search(classifier, data, targets, parameters, 
                                kfold=5, test_size=0.4)


def logistic(data,targets):
    ''' Logistic Regression. (Similar to above).

    Try values of:
    
        C = [0.1, 0.5, 1, 5, 10, 50, 100]

    '''
    # Create parameters
    parameters = [{'C':[0.1, 0.5, 1, 5, 10, 50, 100]}]
    # Create linea regression classifier
    classifier = LogisticRegression()
    # Fit the model using grid search and get the best train and test scores
    return fit_model_grid_search(classifier, data, targets, parameters, 
                                kfold=5, test_size=0.4)

def knn(data,targets):
    ''' k-Nearest Neighbors. (Similar to above).

    Try values of:

        n_neighbors = [1, 2, 3, ..., 50] 
        leaf_size = [5, 10, 15, ..., 60]

    '''
    # Create parameters
    parameters = [{'n_neighbors': list(range(1,51)) ,'leaf_size': list(range(5,61,5))}]
    # Create linea regression classifier
    classifier = KNeighborsClassifier()
    # Fit the model using grid search and get the best train and test scores
    return fit_model_grid_search(classifier, data, targets, parameters, 
                                kfold=5, test_size=0.4)

def decision_tree(data,targets):
    ''' Decision Trees. (Similar to above).

    Try values of:

        max_depth = [1, 2, 3, ..., 50]
        min_samples_split = [2, 3, 4, ..., 10]

    '''
  # Create parameters
    parameters = [{'max_depth': list(range(1,51)) ,'min_samples_split': list(range(2,11))}]
    # Create linea regression classifier
    classifier = DecisionTreeClassifier()
    # Fit the model using grid search and get the best train and test scores
    return fit_model_grid_search(classifier, data, targets, parameters, 
                                kfold=5, test_size=0.4)

def random_forest(data,targets):
    ''' Random Forest. (Similar to above).

    Try values of:
    
         max_depth = [1, 2, 3, ..., 50] 
         min_samples_split = [2, 3, 4, ..., 10]

    '''
  # Create parameters
    parameters = [{'max_depth': list(range(1,51)) ,'min_samples_split': list(range(2,11))}]
    # Create linea regression classifier
    classifier = RandomForestClassifier()
    # Fit the model using grid search and get the best train and test scores
    return fit_model_grid_search(classifier, data, targets, parameters, 
                                kfold=5, test_size=0.4)

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


