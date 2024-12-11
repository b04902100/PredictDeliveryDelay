import os
import glob
import time
from datetime import timedelta
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

def load_data(datafile):
    """
    Read in input file and load data

    return: dataframe
    """

    ## 1. Read in data from input file

    df = pd.read_csv(datafile, header= 0, encoding='unicode_escape')

    print("\n********** Data Summary **********\n")
    print(df.shape, "\n")
    print(df.head(3), "\n")
    print(df.info(), "\n")

    ## 2. Check if any columns contain null values
    print("\n********** Count of Null Values for Each Column **********\n")
    print(df.isnull().sum(), "\n")

    ## 3. Randomly sample 30,000 rows
    sampled_data = df.sample(n=30000, random_state=42)

    print("\n********** Data Shape after Sampling Data **********\n")
    print(sampled_data.shape, "\n")

    return sampled_data


def split_data(X_data, y_data):
    ## 1. Split data

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=5, stratify=y_data)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=5, stratify=y_test)

    ## 2. Reset index

    # Train Data
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # Validation Data
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    # Test Data
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("\n********** Data Shape after Splitting **********\n")
    print("\nX_train: ", X_train.shape)
    print("\nX_val: ", X_val.shape)
    print("\nX_test: ", X_test.shape)

    print("\n********** Data View after Splitting **********\n")
    print("\nX_train:\n", X_train.head(3))
    print("\nX_val:\n", X_val.head(3))
    print("\nX_test:\n", X_test.head(3))

    return (X_train, X_val, X_test, y_train, y_val, y_test)


def fit_model(X, y, modelname):
    """
    This function fits a machine learning model based on the provided model name.
    It supports the following models:
    'DT': Decision Tree
    'LR': Logisitic regression
    'SVM': Support Vector Machines
    'RF': Random Forest

    Parameters:
    X: Feature set (training data)
    y: Target labels (training data)
    modelname: The name of the model to be fitted ('DecisionTree', 'LogisticRegression', 'SVM', or 'RandomForest')

    Returns:
    model: The trained model object
    """
    # Initialize model as None
    model = None

    # Decision Tree
    if modelname == 'DT':
        model = DecisionTreeClassifier()

    # Logistic Regression
    elif modelname == 'LR':
        model = LogisticRegression(max_iter=1000)  # Max iter increased to ensure convergence

    # Support Vector Machine
    elif modelname == 'SVM':
        model = SVC()  # Default SVM (Support Vector Classifier)

    # Random Forest
    elif modelname == 'RF':
        model = RandomForestClassifier()

    # If the model name is not recognized, raise an error
    else:
        raise ValueError(
            "Invalid model name. Choose from 'DecisionTree', 'LogisticRegression', 'SVM', or 'RandomForest'.")

    # Fit the model on the provided training data
    model.fit(X, y)

    print(f"{modelname} model has been fitted successfully.")

    # Return the trained model
    return model


def evaluate_model(y_true, y_pred):
    """
    This function evaluates the model by producing a confusion matrix.

    Parameters:
    y_true: Actual labels
    y_pred: Predicted labels

    Returns:
    None: Displays a confusion matrix and prints raw values.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Non-RCT', 'RCT'],
                yticklabels=['Non-RCT', 'RCT'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Print confusion matrix values
    print("\nConfusion Matrix:")
    print(cm)