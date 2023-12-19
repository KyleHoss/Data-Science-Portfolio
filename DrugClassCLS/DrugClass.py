import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler 

# Injest DataSet
df = pd.read_csv("DrugData.csv")
# print(df.describe())

# Check how many columns have null data
print('Null Values')
print(pd.isnull(df).sum())
print('Duplicated Values')
print(df.duplicated().sum())

# Only catogorial columns
for col in df.select_dtypes(exclude = ['int64', 'float', 'float64', 'int']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

df.drop_duplicates(inplace=True)

# Seperate into Training & Testing Sets
X = df.drop('Drug', axis=1)
y = df['Drug']

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size = 0.3, shuffle = True)

# resample minority class in both test and train
# We do this after the split so that duplicates don't pollute either set
X_train_resamp, y_train_resamp = RandomOverSampler().fit_resample(X_train, y_train)
X_test_resamp, y_test_resamp = RandomOverSampler().fit_resample(X_test, y_test)

# Build the machine learning model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Original Report')
print('-------------------------------')
print(classification_report(y_test, y_pred))
print('                                     ')

model.fit(X_train_resamp, y_train_resamp)
y_pred_resamp = model.predict(X_test_resamp)
print('ReSampled Report')
print('-------------------------------')
print(classification_report(y_test_resamp, y_pred_resamp))