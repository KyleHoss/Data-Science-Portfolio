import time
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
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

#Ingest Dataset
df = pd.read_csv('MortalityInfo.csv')

# Drop Columns that care needed
df= df.drop('ID', axis=1)
df= df.drop('group', axis=1)

# Check how many columns have null data
# print(df.describe())
# print('Null Values')
# print(pd.isnull(df).sum())
# print('')
# print('Duplicated Values:', df.duplicated().sum())

nullColumns = [ 'BMI', 'heart rate', 'Systolic blood pressure',
                'Diastolic blood pressure', 'Respiratory rate','temperature',
                'SP O2', 'Urine output', 'Neutrophils', 'Basophils', 'Lymphocyte', 'Blood calcium',
                'PT', 'INR','Creatine kinase', 'glucose','PH', 'Lactic acid', 'PCO2']

# Since the amount of NULL values is so large I dropped all the rows 
# that the data was not found on

df = df.dropna(axis=0, how='any')
# replacing na values in college with No college\
# for col in nullColumns:
#     df[col].fillna(df[col].mean(), inplace = True)

features = ['age', 'gendera', 'BMI', 'hypertensive',
            'atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias',
            'depression', 'Hyperlipemia', 'Renal failure', 'COPD', 'heart rate',
            'Systolic blood pressure', 'Diastolic blood pressure',
            'Respiratory rate', 'temperature', 'SP O2', 'Urine output',
            'hematocrit', 'RBC', 'MCH', 'MCHC', 'MCV', 'RDW', 'Leucocyte',
            'Platelets', 'Neutrophils', 'Basophils', 'Lymphocyte', 'PT', 'INR',
            'NT-proBNP', 'Creatine kinase', 'Creatinine', 'Urea nitrogen',
            'glucose', 'Blood potassium', 'Blood sodium', 'Blood calcium', 
            'Chloride', 'Anion gap', 'Magnesium ion', 'PH', 'Bicarbonate',
            'Lactic acid', 'PCO2', 'EF']

X_train, X_test, y_train, y_test = train_test_split(df[features],df['outcome'], 
                                                    test_size=0.3,random_state = 0)
print('Null Values')
print(pd.isnull(df).sum())


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