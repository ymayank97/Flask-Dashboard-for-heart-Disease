#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split #for data splitting
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
#reading dataset
dataset=pd.read_csv("C:\\Users\\u22v03\\Documents\\Python Scripts\\heart\\heart.txt",header=None,sep=' ')
dataset.columns=['Age', 'Gender', 'CP', 'Trestbps', 'Chol', 'FBS', 'RestECG',
                 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'CA', 'Thal', 'Goal']

     
dataset['Gender']=dataset['Gender'].replace([1,0], ['Male', 'Female'])
dataset['Goal']=dataset['Goal'].replace([1,2], ['Absence', 'Presence'])
dataset['Slope']=dataset['Slope'].replace([1,2,3], ['Upsloping','Flat','Down-sloping'])
dataset['RestECG']=dataset['RestECG'].replace([0,1,2], ['Normal', 'Abnormality','Hypertrophy'])
dataset['Exang']=dataset['Exang'].replace([1,0], ['Yes', 'No'])
dataset['FBS']=dataset['FBS'].replace([1,0], ['Yes', 'No'])
dataset['Thal']=dataset['Thal'].replace([3,6,7], ['Normal', 'Fixed Defect','Reversible defect'])
dataset['CP']=dataset['CP'].replace([1,2,3,4], ['Typical angina', 'Atypical angina','Non-anginal pain','Asymptomatic pain'])
dataset['Gender']=dataset['Gender'].astype('object')
dataset['CP']=dataset['CP'].astype('object')
dataset['Thal']=dataset['Thal'].astype('object')
dataset['FBS']=dataset['FBS'].astype('object')
dataset['Exang']=dataset['Exang'].astype('object')
dataset['RestECG']=dataset['RestECG'].astype('object')
dataset['Slope']=dataset['Slope'].astype('object')
dataset['Goal']=dataset['Goal'].replace( ['Absence', 'Presence'],[0,1])
dataset['Goal']=dataset['Goal'].astype('int64')
dataset = pd.get_dummies(dataset,drop_first=False)
dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset)).values

X_train, X_test, y_train, y_test = train_test_split(dataset.drop('Goal', 1), dataset['Goal'], test_size = .2, random_state=42,shuffle=True)

#Logistic Regression :- fitting the model
lr=LogisticRegression(C=0.2,random_state=42,penalty='l1',class_weight={0:1,1:1})
lr.fit(X_train,y_train)
joblib.dump(lr, 'model1.pkl')


model = DecisionTreeClassifier(max_depth=6,random_state=42,criterion='entropy',max_features='auto')
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')


#  Random forest  :- fitting the model
model2 = RandomForestClassifier(max_depth=5,oob_score=True,random_state=42,criterion='gini',max_features='auto',n_estimators=300)
model2.fit(X_train,y_train)
joblib.dump(model2, 'model2.pkl')



# Saving the data columns from training
model_columns = list(X_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")