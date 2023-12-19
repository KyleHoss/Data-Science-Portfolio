# Stoke Analysis
from itertools import count
import time
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Import data into Dataframe
df = pd.read_csv("LungCancerInfo.csv")
print(df.info())

# Encode the data to 1 or 0 
for col in df:
    if col == 'AGE':
        continue
    df[col]=LabelEncoder().fit_transform(df[col])

print(df.head())

# Separate independent and dependent attributes
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# Test_train_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)
print(f'Train shape : {X_train.shape}\nTest shape: {X_test.shape}')

# resample minority class in both test and train
# We do this after the split so that duplicates don't pollute either set
X_train_resamp, y_train_resamp = RandomOverSampler().fit_resample(X_train, y_train)
X_test_resamp, y_test_resamp = RandomOverSampler().fit_resample(X_test, y_test)

classifiers = [ KNeighborsClassifier(),         
                SVC(),                          
                #GaussianProcessClassifier(),    
                DecisionTreeClassifier(),       
                RandomForestClassifier(random_state=0, max_depth=5),       
                MLPClassifier(),                
                AdaBoostClassifier(),           
                GaussianNB(),                   
                QuadraticDiscriminantAnalysis(),
                XGBClassifier(),                
                LGBMClassifier(),
                LogisticRegression()]              

names = [ 'KNeighborsClassifier',         
          'SVC',                          
          #'GaussianProcessClassifier',    
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
    st = time.time()
    model = classifiers[i]
    model.fit(X_train_resamp, y_train_resamp)
    y_pred_resamp = model.predict(X_test_resamp)

    print(i," ", names[i])
    print('===============================')
    print(classification_report(y_test_resamp, y_pred_resamp))
    accSamp.insert(i,accuracy_score(y_test_resamp, y_pred_resamp).round(2))
    
    model = classifiers[i]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('Original Data')
    print('===============================')
    print(classification_report(y_test, y_pred))
    accOrg.insert(i,accuracy_score(y_test, y_pred).round(2))

    et = time.time()
    tt = et - st
    tim.insert(i,et - st)

print("Final resSamp--------------------------------------------")
print('Resampled')
resSamp['Model'] = np.array(names)
resSamp['Accuracy'] = np.array(accSamp)
resSamp['Time'] = np.array(tim)
print(resSamp.sort_values(by='Accuracy', ascending=False))

print('Original')
resOrg['Model'] = np.array(names)
resOrg['Accuracy'] = np.array(accOrg)
resOrg['Time'] = np.array(tim)
print(resOrg.sort_values(by='Accuracy', ascending=False))