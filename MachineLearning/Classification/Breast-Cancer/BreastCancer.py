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

#Ingest Dataset
df = pd.read_csv('breast-cancer.csv')

# Drop Columns that care needed
df =df.drop('id', axis=1)

#print(df.info())

# Check how many columns have null data
# print('Null Values')
# print(pd.isnull(df).sum())
# print('')
# print('Duplicated Values:', df.duplicated().sum())

# replacing na values in college with No college
#df["bmi"].fillna(df['bmi'].mean(), inplace = True)
#df.drop_duplicates(inplace=True)

print(df.columns)

# Only catogorial columns
for col in df.select_dtypes(exclude = ['int64', 'float', 'float64', 'int']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

features= [ 'radius_mean', 'texture_mean', 'perimeter_mean',
            'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 
            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',     
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 
            'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave points_worst',       
            'symmetry_worst', 'fractal_dimension_worst']

# Build testing and training Dataset
X_train, X_test, y_train, y_test = train_test_split(df[features],df['diagnosis'], 
                                                    test_size=0.3,random_state = 0)

# resample minority class in both test and train
# We do this after the split so that duplicates don't pollute either set
X_train_resamp, y_train_resamp = RandomOverSampler().fit_resample(X_train, y_train)
X_test_resamp, y_test_resamp = RandomOverSampler().fit_resample(X_test, y_test)

classifiers = [ KNeighborsClassifier(),         
                SVC(kernel='poly'),                          
                #GaussianProcessClassifier(),    
                DecisionTreeClassifier(),       
                RandomForestClassifier(),       
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

    print(i,"Resampled Data", names[i])
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