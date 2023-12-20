# Data Manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Silence Warnings
import time
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Algorithms
from sklearn.svm import SVC
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

#---DATA PREPROCESSING###

# Ingest into DataFrame
df_train = pd.read_csv('diabetes.csv')

# Check for Null Values
print('Null Values:', pd.isnull(df_train).sum())

# Check for Duplicated Values
print('Duplicated Values:', df_train.duplicated().sum())

# Remove Null values
df_train = df_train.dropna()

# Remove Unnessarcy Columns
    # Don't really need to remove anything.

# Encode Data
    # Don't have to encode anything. 

# Seperate into Training and Testing Sets
X= df_train.drop(columns=['Outcome'])
y= df_train['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state = 0)

# resample minority class in both test and train
X_train_resamp, y_train_resamp = RandomOverSampler().fit_resample(X_train, y_train)
X_test_resamp, y_test_resamp = RandomOverSampler().fit_resample(X_test, y_test)


#---MACHINE LEARNING

# Machine Learning models
classifiers = [ KNeighborsClassifier(),         
                SVC(kernel='rbf'),                          
                GaussianProcessClassifier(),
                GradientBoostingClassifier(random_state=0),    
                DecisionTreeClassifier(),       
                RandomForestClassifier(random_state=0),       
                MLPClassifier(),                
                AdaBoostClassifier(),           
                GaussianNB(),                   
                QuadraticDiscriminantAnalysis(),
                XGBClassifier(),                
                LGBMClassifier(random_state=0),
                LogisticRegression()]    
         
# Machine Learning Names
names = [ 'KNeighborsClassifier',         
          'SVC',                          
          'GaussianProcessClassifier', 
          'GradientBoosingClassifier',    
          'DecisionTreeClassifier',       
          'RandomForestClassifier',       
          'MLPClassifier',                
          'AdaBoostClassifier',           
          'GaussianNB',                   
          'QuadraticDiscriminantAnalysis',
          'XGBClassifier',                
          'LGBMClassifier',
          'LogisticRegression'] 
          
accSamp = [] * len(classifiers)
accOrg = [] * len(classifiers)
tim = [] * len(classifiers)
resSamp = pd.DataFrame()
resOrg = pd.DataFrame()

for i in range(0, len(classifiers)):
    # Start time
    st = time.time()

    # Resample Model Predicitions
    model = classifiers[i]
    model.fit(X_train_resamp, y_train_resamp)
    y_pred_resamp = model.predict(X_test_resamp)
    accSamp.insert(i,accuracy_score(y_test_resamp, y_pred_resamp).round(2))
    
    # Original Model Predictions
    model = classifiers[i]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accOrg.insert(i,accuracy_score(y_test, y_pred).round(2))

    # End Time
    et = time.time()
    tim.insert(i,et - st)

print("Final resSamp--------------------------------------------")
print('Resampled')
resSamp['Model'] = np.array(names)
resSamp['Accuracy'] = np.array(accSamp)
resSamp['Time'] = np.array(tim)
print(resSamp.sort_values(by='Accuracy', ascending=False))

print('')
print('Original')
resOrg['Model'] = np.array(names)
resOrg['Accuracy'] = np.array(accOrg)
resOrg['Time'] = np.array(tim)
print(resOrg.sort_values(by='Accuracy', ascending=False))