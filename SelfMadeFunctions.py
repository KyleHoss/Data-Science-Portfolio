'''
Self Made Functions-
I tend to use these functions alot so I have imported them for easier use. :-)
'''

# Import Core Libraries
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import Metric & Preprocessing Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, recall_score
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report

# IMPORTANCE FUNCTIONS
def treeRegressionImportance(model_name, model, X_train, y_train):
    '''
    Title: Tree Regression Importance
    Description: This will output features of importance in a tree regression model.    
                 Random Forest & Decision Tree Importance.

    model_name: The name of the tree model.
    model: Tree Model
    X_train: X_train dataset.
    y_train: y_train dataset.
    top_values: The top # of features of importance.
    '''
    # Create title for Chart
    title = 'Feature Importance for '+ model_name

    # Fit model
    model.fit(X_train, y_train)

    # Create list to hold features values
    feature_list = []

    # Extract feature names and their importance
    for feature, importance in zip(X_train.columns, model.feature_importances_):  
        temp = [feature, importance]
        feature_list.append(temp)

    # Create Pandas DataFrame
    importance_df= pd.DataFrame(feature_list, columns = ['Feature', 'Importance'])
    importance_df= importance_df.sort_values('Importance', ascending = False)

    # Display Importance DataFrame
    display(importance_df)

    # Create Barchart
    sns.barplot(data=importance_df,
                x='Importance',
                y='Feature',
                orient='h')

    # Title for chart
    plt.title(title)

def logisticRegressionImportance(model, X, X_train, y_train): 
    '''
    Title: Logistic Regression Importance
    Description: This will display the importances of a Logisitic Regression Model.
    
    model: Logistic Regression Model
    X: X features.
    X_train: X_train array.
    y_train: y_train array.
    '''
    # Assign model to variable
    log_model = model

    # Train Machine Learning Model
    log_model.fit(X_train, y_train)
    
    # Importance Information
    results = {
        'Feature': X.columns,
        'Scoring': abs(log_model.coef_[0])
    }
    
    # Put importance info into dataframe
    df_Scores = pd.DataFrame(results)
    
    # Order DataFrame
    df_Scores = df_Scores.sort_values(by='Scoring', ascending=False)
    df_Scores = df_Scores.reset_index()
    print(' '*2,'Logisitic Regression Feature Importance')
    print('-'*40)
    display(df_Scores)
    
    # Figure Size
    plt.figure(figsize=(12,8))
    
    # Plot importance of features
    sns.barplot(data=df_Scores, 
                y='Feature', 
                x='Scoring', 
                orient='h')
    
    # Title of figures
    plt.title('Logisitic Regression Feature Importance')

def xgbImportance(model, X, X_train, y_train): 
    '''
    Title: XGB Importance
    Description: This will display the importances of a Logisitic Regression Model.
    
    model: XGB Model
    X: X features.
    X_train: X_train array.
    y_train: y_train array.
    '''
    # Assign model to variable
    xgb_model = model

    # Train Machine Learning Model
    xgb_model.fit(X_train, y_train)
    
    # Importance Information
    results = {
        'Feature': X.columns,
        'Scoring': xgb_model.feature_importances_
    }
    
    # Put importance info into dataframe
    df_Scores = pd.DataFrame(results)
    
    # Order DataFrame
    df_Scores = df_Scores.sort_values(by='Scoring', ascending=False)
    print(' '*2,'XGB Classifer Feature Importances')
    print('-'*40)
    display(df_Scores)
    
    # Figure Size
    plt.figure(figsize=(12,8))
    
    # Plot importance of features
    sns.barplot(data=df_Scores, 
                y='Feature', 
                x='Scoring', 
                orient='h')
    
    # Title of figures
    plt.title('XGB Classifier Feature Importance')


# VISUALIZATION FUNCTIONS
def histogram_boxplot(data, feature, figsize = (12, 7), kde = False):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    """
    # Title of Chart
    title = 'Univariate Analysis of ' + feature

    # Outlier Calculations
    upper_outlier = data[feature].mean() + data[feature].std()*3
    lower_outlier = data[feature].mean() - data[feature].std()*3

    # Structure of Chart
    f2, (ax_box2, ax_hist2) = plt.subplots( nrows = 2,                                        # Number of rows of the subplot grid = 2
                                            sharex = True,                                    # x-axis will be shared among all subplots
                                            gridspec_kw = {"height_ratios": (0.25, 0.75)},    # Ratios of hieght
                                            figsize = figsize)                                # Creating the 2 subplots
    
    # Add title to figure
    plt.suptitle(title) 
    
    # Plot Boxplot 
    sns.boxplot( data = data, 
                 x = feature, 
                 ax = ax_box2, 
                 showmeans = True, 
                 color = "violet")  
    
    # Plot Histogram
    sns.histplot(data = data, 
                 x = feature, 
                 kde = kde, 
                 ax = ax_hist2)
    
    # Add Mean
    ax_hist2.axvline(data[feature].mean(), 
                     color = "green", 
                     linestyle = "--")  
    
    # Add Median
    ax_hist2.axvline(data[feature].median(), 
                     color = "black", 
                     linestyle = "-")  
    
    # Add Upper Outlier
    ax_hist2.axvline(upper_outlier, 
                     color = "red", 
                     linestyle = "-")

    # Add Lower Outlier
    ax_hist2.axvline(lower_outlier, 
                     color = "red", 
                     linestyle = "-")  

def categoricalVisualizations(figsize, title, rows, columns, dataFrame, cat_columns):
  '''
  Title: Categorical Countplot Visualizations
  Description: This is create a figure with subplots for distribution of
               categorical columns.

  figsize= Height and width of figure size.
  title= Title of figure.
  rows= Number of rows in figure.
  columns= Number of columns in figure. 
  dataFrame= DataFrame of values.
  cat_columns= List of categorical columns.
  '''
  # Choose Size of Chart
  plt.figure(figsize=figsize)
  plt.suptitle(title)

  for i in range(0,len(cat_columns)):
    
    # Iterate through subplots
    plt.subplot(rows,columns,i+1)

    # Plot CountPlot
    ax = sns.countplot(data=dataFrame,                                                                    # DataFrame of values
                       x=dataFrame[cat_columns[i]],                                                       # Column Name
                       order=dataFrame[cat_columns[i]].value_counts().sort_values(ascending=True).index)  # Order of values

    # Value coutns for each bar
    abs_values = dataFrame[cat_columns[i]].value_counts(ascending=True).values

    ax.bar_label(container=ax.containers[0], labels=abs_values)

def continuousVisulaizations(figsize, title, rows, columns, dataFrame, con_columns):
  '''
  Title: Continuous Distribution Visualizations
  Description: This is create a figure with subplots for distribution of
               continuous columns.

  figsize= Height and width of figure size.
  title= Title of figure.
  rows= Number of rows in figure.
  columns= Number of columns in figure. 
  dataFrame= DataFrame of values.
  con_columns= List of continuous column names.
  '''
  # Choose Size of Chart
  plt.figure(figsize=figsize)
  plt.suptitle(title)

  for i in range(0,len(con_columns)):

    # Statisical Computations
    mean = dataFrame[con_columns[i]].mean()     # Calculate mean for column
    median = dataFrame[con_columns[i]].median() # Calculate median for column

    # Calculate lower and higher outlier range
    low_outliers = mean - dataFrame[con_columns[i]].std()*3   # Calcualte lower outliers     
    high_outliers = mean + dataFrame[con_columns[i]].std()*3  # Calculate upper ouliers

    # Iterate through subplots
    plt.subplot(rows,columns,i+1)

    # Plot Histograms
    sns.histplot(data=dataFrame,    # DataFrame  
                 x=con_columns[i],  # Column Name
                 kde=True)          # Show KDE Distribution

    # If outliers are below 0 do no show lower outlier line
    if low_outliers > 0:
      plt.axvline(low_outliers, color='r')

    # Plot Statisical Metrics on chart
    plt.axvline(mean, color='g')
    plt.axvline(median, color='orange')
    plt.axvline(high_outliers, color='r')

def precisionRecallAnalysis(y_true, y_proba):
  '''
  Title: Precision & Recall Analysis
  Description: This will produce two charts that to give visualization to 
               precision and recall.
  
  y_true: The true values target values.
  y_proba: The associated probabilies for making predicitons on y_true. 
  '''
  # Precision Recall Analysis 
  precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba[:,1])

  # Set size of graph
  plt.figure(figsize=(16, 7))  
  plt.suptitle('Precision vs. Recall Analysis')

  # Create First plot
  plt.subplot(1,2,1)
  plt.title('Precision & Recall VS. Threshold')
  plt.xlabel("Threshold")
  plt.ylabel('Precision & Recall')
  line1 = sns.lineplot(x = thresholds, y = precisions[:-1], color='b', lw=2)
  line2 = sns.lineplot(x = thresholds , y = recalls[:-1], color='g', lw=2)
  
  # Set Lines to dashes
  line1.lines[0].set_linestyle("--")
  line1.lines[1].set_linestyle("--")
  plt.legend(loc='lower left', labels=['Precision', 'Recall'])

  # Create Presicion VS. Recall Plot
  plt.subplot(1,2,2)
  plt.title('Precision VS. Recall')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  sns.lineplot(x=recalls[:-1], y=precisions[:-1], lw=2)


# MACHINE LEARNING FUNCTIONS
def classifierMetrics(classifiers,classifier_name,X, y, X_train, y_train, X_test, y_test, kfolds):
  import warnings
  warnings.filterwarnings('ignore')
  '''
  Title: Metrics for Classification
  Description: This will perform a varity of Classification Algorithms 
               on the provided datasets. 

  classifiers: Lists of classifiers
  classifier_name: Name of classifiers.
  X: X_dataframe
  y: y_dataframe
  X_train: Array of training features.
  y_train: Array of target values. 
  X_test: Testing feature array for validation.
  y_test: Testing value array for validation.
  '''

  # DataFrame of Trained Results
  test_results = pd.DataFrame()
  cross_results = pd.DataFrame()
  
  # Arrays for Training Results
  test_auc = [] * len(classifiers)
  test_recall = [] * len(classifiers)
  test_accuracy = [] * len(classifiers)
  test_f1_score = [] * len(classifiers)
  test_precision = [] * len(classifiers)

  # Arrays for Test Results
  cross_f1 = [] * len(classifiers)
  cross_auc = [] * len(classifiers)
  cross_recall = [] * len(classifiers)
  cross_accuracy = [] * len(classifiers)
  cross_precision = [] * len(classifiers)

  # Loop through all classifiers and find best performing alogrithm
  for i in range(0,len(classifiers)):

    # Assign and Start time
    trained_model = classifiers[i]   # Assign Model  

    # Train using Cross Validation
    cv_results = cross_validate(classifiers[i],X,y,cv=kfolds, scoring=['accuracy', 'f1','precision', 'recall', 'roc_auc'])

    # Predict Test Data
    trained_model.fit(X_train, y_train)
    y_test_pred = trained_model.predict(X_test)

    # Insert Training Metrics into arrays
    cross_f1.insert(i, cv_results['test_f1'][:-1].mean())
    cross_auc.insert(i, cv_results['test_roc_auc'][:-1].mean())
    cross_recall.insert(i, cv_results['test_recall'][:-1].mean())
    cross_accuracy.insert(i, cv_results['test_accuracy'][:-1].mean())
    cross_precision.insert(i, cv_results['test_precision'][:-1].mean())

    # Insert Testing Metrics into arrys
    test_auc.insert(i, roc_auc_score(y_test, y_test_pred))
    test_recall.insert(i, recall_score(y_test, y_test_pred))
    test_accuracy.insert(i, accuracy_score(y_test, y_test_pred))
    test_f1_score.insert(i, f1_score(y_test, y_test_pred))
    test_precision.insert(i, precision_score(y_test, y_test_pred))

  # Add data into Training DataFrame
  cross_results['Model'] = np.array(classifier_name)
  cross_results['Recall'] = np.array(cross_recall)
  cross_results['Precision'] = np.array(cross_precision)
  cross_results['AUC'] = np.array(cross_auc)
  cross_results['F1_Score'] = np.array(cross_f1)
  cross_results['Accuracy'] = np.array(cross_accuracy)

  # Add data into Testing DataFrame 
  test_results['Model'] = np.array(classifier_name)
  test_results['Recall'] = np.array(test_recall)
  test_results['Precision'] = np.array(test_precision)
  test_results['AUC'] = np.array(test_auc)
  test_results['F1_Score'] = np.array(test_f1_score)
  test_results['Accuracy'] = np.array(test_accuracy)

  # Order Training Results
  cross_results = cross_results.sort_values(by='F1_Score', ascending=False)

  # Order Testing Results
  test_results = test_results.sort_values(by='F1_Score', ascending=False)

  # Print Results for training
  print('Cross-Validation Metrics')
  display(cross_results)

  # Print Results for Testing 
  print('\n')
  print('Train & Test Metrics')
  display(test_results)

def regressionMetrics(regressors, regressor_name, X, y, X_train, y_train, X_test, y_test, kfolds):
  import warnings
  warnings.filterwarnings('ignore')
  '''
  Description: This will perform a varity of Classification Algorithms 
               on the provided datasets. 
  X: X_dataframe
  y: y_dataframe
  X_train: Array of training features.
  y_train: Array of target values. 
  X_test: Testing feature array for validation.
  y_test: Testing value array for validation.
  '''

  # DataFrame of Trained Results
  test_results = pd.DataFrame()
  cross_results = pd.DataFrame()
  
  # Arrays for Training Results
  test_auc = [] * len(classifiers)
  test_recall = [] * len(classifiers)
  test_accuracy = [] * len(classifiers)
  test_f1_score = [] * len(classifiers)
  test_precision = [] * len(classifiers)

  # Arrays for Test Results
  cross_f1 = [] * len(classifiers)
  cross_auc = [] * len(classifiers)
  cross_recall = [] * len(classifiers)
  cross_accuracy = [] * len(classifiers)
  cross_precision = [] * len(classifiers)

  # Loop through all classifiers and find best performing alogrithm
  for i in range(0,len(classifiers)):

    # Assign and Start time
    trained_model = classifiers[i]   # Assign Model  

    # Train using Cross Validation
    cv_results = cross_validate(classifiers[i],X,y,cv=kfolds, scoring=['accuracy', 'f1','precision', 'recall', 'roc_auc'])

    # Predict Test Data
    trained_model.fit(X_train, y_train)
    y_test_pred = trained_model.predict(X_test)

    # Insert Training Metrics into arrays
    cross_f1.insert(i, cv_results['test_f1'][:-1].mean())
    cross_auc.insert(i, cv_results['test_roc_auc'][:-1].mean())
    cross_recall.insert(i, cv_results['test_recall'][:-1].mean())
    cross_accuracy.insert(i, cv_results['test_accuracy'][:-1].mean())
    cross_precision.insert(i, cv_results['test_precision'][:-1].mean())

    # Insert Testing Metrics into arrys
    test_auc.insert(i, roc_auc_score(y_test, y_test_pred))
    test_recall.insert(i, recall_score(y_test, y_test_pred))
    test_accuracy.insert(i, accuracy_score(y_test, y_test_pred))
    test_f1_score.insert(i, f1_score(y_test, y_test_pred))
    test_precision.insert(i, precision_score(y_test, y_test_pred))

  # Add data into Training DataFrame
  cross_results['Model'] = np.array(classifier_name)
  cross_results['Recall'] = np.array(cross_recall)
  cross_results['Precision'] = np.array(cross_precision)
  cross_results['AUC'] = np.array(cross_auc)
  cross_results['F1_Score'] = np.array(cross_f1)
  cross_results['Accuracy'] = np.array(cross_accuracy)

  # Add data into Testing DataFrame 
  test_results['Model'] = np.array(classifier_name)
  test_results['Recall'] = np.array(test_recall)
  test_results['Precision'] = np.array(test_precision)
  test_results['AUC'] = np.array(test_auc)
  test_results['F1_Score'] = np.array(test_f1_score)
  test_results['Accuracy'] = np.array(test_accuracy)

  # Order Training Results
  cross_results = cross_results.sort_values(by='F1_Score', ascending=False)

  # Order Testing Results
  test_results = test_results.sort_values(by='F1_Score', ascending=False)

  # Print Results for training
  print('Cross-Validation Metrics')
  display(cross_results)

  # Print Results for Testing 
  print('\n')
  print('Train & Test Metrics')
  display(test_results)


# USEFUL FUNCTIONS
def seriesStandardScaler(dataFrame, column):
  '''
  Description: This will standarize and scale the column of a dataframe. 

  dataFrame= The dataframe that this column is found in. 
  column= The column of the dataframe that needs to be standardized and scaled. 
  '''
  # Compute Statistics
  mean = dataFrame[column].mean()
  standard_deviation = dataFrame[column].std()

  # Standardize Column
  dataFrame[column] = (dataFrame[column] - mean) / standard_deviation
  return dataFrame[column]

def infoOut(data):
    '''
    Title: Display Dataframe of df.info
    Description: Display a pandas dataframe of df.info

    data: Pandas dataframe.
    '''
    dfInfo = data.columns.to_frame(name='Column')   # Create dataframe         
    dfInfo['Non-Null Count'] = data.notna().sum()   # Add non-null counts to dataframe
    dfInfo['NULL Count'] = data.isnull().sum()      # Add NULL counts to dataframe        
    dfInfo['Dtype'] = data.dtypes                   # add dtype to dataframe
    dfInfo.reset_index(drop=True,inplace=True)      # Reset index        
    return dfInfo                                   # display info dataframe

def nullValues(data):
    '''
    Title: Display Null values in Pandas Dataframe
    Description: Display a pandas dataframe of Null values for each column
    
    data: Pandas dataframe.
    '''
    display(data.isnull().sum().to_frame().rename(columns = {0:'NULL Amounts'}))