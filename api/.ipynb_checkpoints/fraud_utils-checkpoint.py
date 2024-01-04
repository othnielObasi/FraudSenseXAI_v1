#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
from yellowbrick.classifier import ClassificationReport, ClassPredictionError, ROCAUC
import joblib


def load_fraud_data(file_path):
    """
    Loads fraud data from a CSV file into a Pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pandas.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
    
    


def split_and_sample_data(df, target_column, sampling_technique, random_state=None): 
    
    """
    Prepares the data for modeling by dropping specified columns, splitting the data,
    and applying the chosen sampling technique.

    Parameters:
    - df (DataFrame): The input dataframe.
    - columns_to_drop (list): List of columns to drop from the dataframe.
    - target_column (str): The name of the target column.
    - sampling_technique (str): The sampling technique to use ('adasyn', 'smote', or 'smotetomek').

    Returns:
    - X_train, X_test, y_train, y_test: Training and test sets.
    """
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data into training and testing sets with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    

    # Applying the chosen sampling technique
    if sampling_technique.lower() == 'adasyn':
        sampler = ADASYN(sampling_strategy='minority', random_state=random_state)
    elif sampling_technique.lower() == 'smote':
        sampler = BorderlineSMOTE(random_state=random_state)
    elif sampling_technique.lower() == 'smotetomek':
        sampler = SMOTETomek(random_state=random_state)
    else:
        raise ValueError("Invalid sampling technique. Choose 'adasyn', 'smote', or 'smotetomek'.")

    # Create a pipeline for resampling
    pipeline = Pipeline([('sampler', sampler)])
    X_train_resampled, y_train_resampled =  pipeline.named_steps['sampler'].fit_resample(X_train, y_train)


    return X_train_resampled, X_test, y_train_resampled, y_test



def create_pipeline(model):
    """
    Creates a data pipeline with scaling for numeric features followed by a machine learning model.

    Parameters:
    - model: The pre-initialized machine learning model to be used in the pipeline.

    Returns:
    - Pipeline: A sklearn Pipeline object.
    """
    # Define the scaler for numeric features
    transformer = StandardScaler()

    # Create the final pipeline with the scaler and model
    pipeline = Pipeline([
        ('scaler', transformer),
        ('model', model)
    ])

    return pipeline




def cross_validation(pipeline, X_train, y_train, n_splits=10):
    """
    Evaluates a pre-initialized pipeline using cross-validation and calculates performance metrics.

    Parameters:
    - pipeline: The pre-initialized machine learning pipeline to be evaluated, including preprocessing and model.
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.
    - n_splits (int): Number of splits for cross-validation.

    Returns:
    - DataFrame: A DataFrame containing the precision, recall, F1 score, ROC AUC, and accuracy for each fold.
    """

    # Define the cross-validator
    cv = StratifiedKFold(n_splits=n_splits)

    # Store the metrics for each fold
    metrics = {'Precision': [], 'Recall': [], 'F1 Score': [], 'ROC AUC': [], 'Accuracy': []}

    # Perform cross-validation
    for train_index, test_index in cv.split(X_train, y_train):
        # Split data for this fold
        X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]

        # Train the pipeline on the fold's training data
        pipeline.fit(X_fold_train, y_fold_train)

        # Make predictions on the fold's test data
        y_pred_fold = pipeline.predict(X_fold_test)
        y_pred_proba_fold = pipeline.predict_proba(X_fold_test)[:, 1]

        # Calculate metrics for this fold
        metrics['Precision'].append(precision_score(y_fold_test, y_pred_fold, zero_division=0))
        metrics['Recall'].append(recall_score(y_fold_test, y_pred_fold, zero_division=0))
        metrics['F1 Score'].append(f1_score(y_fold_test, y_pred_fold, zero_division=0))
        metrics['ROC AUC'].append(roc_auc_score(y_fold_test, y_pred_proba_fold))
        metrics['Accuracy'].append(accuracy_score(y_fold_test, y_pred_fold))

    # Convert metrics to a DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Calculate mean and standard deviation and add them to the DataFrame
    metrics_df.loc['Mean'] = metrics_df.mean()
    metrics_df.loc['Std'] = metrics_df.std()

    return metrics_df



def plot_confusion_matrix(pipeline, X_test, y_test):
    """
    Generates a heatmap for the confusion matrix of the given true labels and predictions.

    Parameters:
    - y_true: Array-like, true labels.
    - y_pred: Array-like, predicted labels.
    - title: Title for the confusion matrix plot.
    - xticklabels: Labels for the x-axis (default is ["Fraud", "Legit"]).
    - yticklabels: Labels for the y-axis (default is ["Fraud", "Legit"]).
    - figsize: Tuple, size of the figure (default is (8, 6)).
    - cmap: Colormap for the heatmap (default is 'Blues').
    """
    # Calculate the confusion matrix
    y_pred = pipeline.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdPu', cbar=True,
                xticklabels=["Legit", " Fraud"], yticklabels=["Legit", " Fraud"])
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix')
    plt.show()





def evaluate_model(pipeline, X_test, y_test):
    """
    Make predictions on the test set using the given model and evaluate its performance.

    Parameters:
    - model: The trained machine learning model.
    - X_test: Test features.
    - y_test: True labels for the test data.

    Returns:
    - classification_report_output: Classification report as a string.
    - roc_auc: ROC AUC score.
    """

    # Predictions on the test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate the classification report
    classification_report_output = classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"])

    # Calculate the ROC AUC score
    roc_auc_output = roc_auc_score(y_test, y_pred_proba)

    # Print the classification report and ROC AUC score
    print("Classification Report:")
    print(classification_report_output)
    print("ROC AUC Score:", (roc_auc_output).round(6))

    return classification_report_output, roc_auc_output




import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_roc_curve(pipeline, X_test, y_test):
    """
    Plots the ROC curve for a binary classification model.

    Parameters:
    - y_true: Array-like, true binary labels (0 or 1).
    - y_pred_proba: Array-like, predicted probabilities for the positive class.

    Returns:
    - None (displays the ROC curve plot).
    """
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    #lt.grid(True)
    plt.show()
    

def plot_precision_recall_curve(pipeline, X_test, y_test):
    """
    Plots the Precision-Recall curve for a binary classification model.

    Parameters:
    - y_true: Array-like, true binary labels (0 or 1).
    - y_pred_proba: Array-like, predicted probabilities for the positive class.

    Returns:
    - None (displays the Precision-Recall curve plot).
    """
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

def plot_precision_recall_vs_threshold(pipeline, X_test, y_test):
    """
    Plots Precision and Recall versus Threshold for a binary classification model.

    Parameters:
    - y_true: Array-like, true binary labels (0 or 1).
    - y_pred_proba: Array-like, predicted probabilities for the positive class.

    Returns:
    - None (displays the Precision and Recall versus Threshold plot).
    """
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1],"b--", label='Precision')
    plt.plot(thresholds, recall[:-1],"g-", label='Recall', linewidth=2)
    plt.xlabel('Threshold')
    plt.legend(loc='best')
    plt.title('Precision and Recall vs Threshold')
    plt.grid(True)
    plt.show()





def plot_precision_recall_vs_threshold_with_tradeoff(y_true, y_pred_proba):
    """
    Plots Precision and Recall versus Threshold for a binary classification model and marks the trade-off point.

    Parameters:
    - y_true: Array-like, true binary labels (0 or 1).
    - y_pred_proba: Array-like, predicted probabilities for the positive class.

    Returns:
    - None (displays the Precision and Recall versus Threshold plot with the trade-off point marked).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)  # Calculate F1 scores

    # Find the threshold with the highest F1 score (trade-off point)
    tradeoff_threshold = thresholds[np.argmax(f1_scores)]
    tradeoff_precision = precision[np.argmax(f1_scores)]
    tradeoff_recall = recall[np.argmax(f1_scores)]

    plt.figure(figsize=(8, 6))
    plt.plot(precision, recall, linewidth=2)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall')

    # Mark the trade-off point
    plt.scatter([tradeoff_precision], [tradeoff_recall], c='red', marker='o', s=100, label='Trade-off Point')

    # Add dotted lines from both sides to the trade-off point
    plt.axvline(tradeoff_precision, color='gray', linestyle='--')
    plt.axhline(tradeoff_recall, color='gray', linestyle='--')

    plt.grid(True)
    plt.legend()
    plt.show()




def display_precision_recall_plot_with_tradeoff(y_true, y_pred_proba):
    """
    Displays the Precision vs Recall plot with the trade-off point marked.

    Parameters:
    - y_true: Array-like, true binary labels (0 or 1).
    - y_pred_proba: Array-like, predicted probabilities for the positive class.

    Returns:
    - None (displays the Precision vs Recall plot with the trade-off point).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)  # Calculate F1 scores

    # Find the threshold with the highest F1 score (trade-off point)
    tradeoff_threshold = thresholds[np.argmax(f1_scores)]
    tradeoff_precision = precision[np.argmax(f1_scores)]
    tradeoff_recall = recall[np.argmax(f1_scores)]

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')

    # Mark the trade-off point
    plt.scatter([tradeoff_recall], [tradeoff_precision], c='red', marker='o', s=100, label='Trade-off Point')

    plt.grid(True)
    plt.legend()
    plt.show()


def tune_and_retrain_pipeline(pipeline, param_grid, X_train, y_train):
    """
    Fine-tunes a pipeline using GridSearchCV and retrains it with the best parameters.

    Parameters:
    - pipeline: sklearn.pipeline.Pipeline
        The pipeline to be fine-tuned and retrained.
    - param_grid: dict
        The parameter grid for GridSearchCV.
    - X_train: pandas.DataFrame or numpy.array
        Training feature data.
    - y_train: pandas.Series or numpy.array
        Training target data.

    Returns:
    - tuple (sklearn.pipeline.Pipeline, dict)
        A tuple containing the retrained pipeline and the best parameters.
    """

    # Fine-tuning using GridSearchCV without verbose and n_jobs
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Access the best parameters and retrain the pipeline
    best_params = grid_search.best_params_
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)

    return pipeline, best_params



def plot_classification_report(pipeline, X_train, y_train, X_test, y_test, title="Classification Report"):
    """
    Create and display the Classification Report visualizer.

    Parameters:
    - model: Trained classifier model.
    - X_train: Training feature data.
    - y_train: Training target data.
    - X_test: Test feature data.
    - y_test: Test target data.
    - title: str, optional
        Custom title for the visualization. Default is "Classification Report".
    """
    visualizer = ClassificationReport(pipeline, support=True, title=title, fmt='2f')
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    
    visualizer.poof()

def plot_class_prediction_error(pipeline, X_train, y_train, X_test, y_test, title="Class Prediction Error"):
    """
    Create and display the Class Prediction Error visualizer.

    Parameters:
    - model: Trained classifier model.
    - X_train: Training feature data.
    - y_train: Training target data.
    - X_test: Test feature data.
    - y_test: Test target data.
    - title: str, optional
        Custom title for the visualization. Default is "Class Prediction Error".
    """
    visualizer = ClassPredictionError(pipeline)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.poof()

def plot_roc_auc(pipeline, X_train, y_train, X_test, y_test):
    """
    Create and display the ROCAUC visualizer.

    Parameters:
    - model: Trained classifier model.
    - X_train: Training feature data.
    - y_train: Training target data.
    - X_test: Test feature data.
    - y_test: Test target data.
    - title: str, optional
        Custom title for the visualization. Default is "ROC-AUC Curve".
    """
    visualizer = ROCAUC(pipeline)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.poof()


def get_data():
    df = load_fraud_data('fraud_data.csv')
   
    X_train_resampled_ad, X_test, y_train_resampled_ad, y_test = split_and_sample_data(df, target_column = 'isFraud',
                                                                 sampling_technique='adasyn', random_state=42)
    X_train_resampled_smote, X_test, y_train_resampled_smote, y_test = split_and_sample_data(df, target_column = 'isFraud',
                                                                 sampling_technique='smote', random_state=42)

    X_train_resampled_stomek, X_test, y_train_resampled_stomek, y_test = split_and_sample_data(df, target_column = 'isFraud',
                                                                 sampling_technique='smotetomek', random_state=42)
    return  X_train_resampled_ad, y_train_resampled_ad, X_train_resampled_smote, y_train_resampled_smote, X_train_resampled_stomek, y_train_resampled_stomek






