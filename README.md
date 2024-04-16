# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or sum values using .isnull() and .sum() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Predict the values of array.

5.Calculate the accuracy by importing the required modules from sklearn.

6.Apply new unknown values.

## Program:
```py
```py
'''
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: K KESAVA SAI 
RegisterNumber: 212223230105 
'''
```
import pandas as pd 
data = pd.read_csv('Employee.csv')
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

## Head():
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849303/fe2aeced-7918-45fe-a011-3b42a182846c)
## info():
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849303/315a8c82-7f0c-4eba-936a-b7324d52e413)
## isnull().sum():
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849303/71de2675-e82a-467a-a486-8d176b5c2a9a)
## Left value counts:
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849303/0d0ed822-d019-462f-b5ad-a80d7809c509)
## Head()(After transform of salary):
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849303/2a9e3bcc-6c62-4374-8c7b-cc8e0917b84e)
## After removing left and departments columns:
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849303/d1730f2d-3d06-4bf4-a183-74aa9b4b094e)
## accuracy:
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849303/7e0cd8de-e178-4fb2-acf0-391722f98969)
## prediction:
![image](https://github.com/Kesavasai20/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/138849303/17d35575-122a-4e6e-8894-1213e676db20)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
