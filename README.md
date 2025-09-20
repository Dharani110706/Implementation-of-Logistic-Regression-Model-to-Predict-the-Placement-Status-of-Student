# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:

1. Hardware – PCs
   
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1. Start

Step 2. Load the California Housing dataset.

Step 3. select the first 3 features as input (X) and target variables (Y).

Step 4. Split the data into training and testing sets .

Step 5. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.

Step 6. Make predictions on the test data, inverse transform the predictions.

Step 7. Then Calculate the Mean Squared Error.

Step 8. End

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DHARANI SREE P
RegisterNumber: 212224040071
*/
```

```
import pandas as pd
data=pd.read_csv("place.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data["gender"])
data1["ssc_b"]=le.fit_transform(data["ssc_b"])
data1["hsc_b"]=le.fit_transform(data["hsc_b"])
data1["hsc_s"]=le.fit_transform(data["hsc_s"])
data1["degree_t"]=le.fit_transform(data["degree_t"])
data1["workex"]=le.fit_transform(data["workex"])
data1["specialisation"]=le.fit_transform(data["specialisation"])
data1["status"]=le.fit_transform(data["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#library for large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

## Accuracy:

<img width="248" height="68" alt="364631379-ff5ab1da-b0e1-4dec-bc1b-a921b88d412c" src="https://github.com/user-attachments/assets/04b4e077-e5d0-4258-8613-7633ce6722f4" />

## Classification report value:

<img width="677" height="245" alt="364631426-5444e1de-0272-4bf9-b924-6c1cbb152ca4" src="https://github.com/user-attachments/assets/38ad4e8d-86ec-47f4-b43d-20be954844f9" />

## Predict value:

<img width="177" height="47" alt="364631464-019cd29c-d967-402a-b900-17aa812f2090" src="https://github.com/user-attachments/assets/080a5d83-bba3-420a-8676-1d8cea77d959" />

## Result:

Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
