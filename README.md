# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the employee dataset, handle missing values, and encode categorical columns.
2. Split the data into features (X) and target (y), then into training and testing sets.
3. Train a DecisionTreeClassifier on the training data.
4. Predict outcomes on test data, calculate accuracy, and visualize the decision tree.

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Venkatachalam S
RegisterNumber:  212224220121

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv('/content/Employee.csv')

print("Data head:")
df.head()

print("Data info:")
df.info()

print("Data null values:")
df.isnull().sum()

print("Data value counts:")
df['left'].value_counts()

le = LabelEncoder()
print("Data head salary:")
df['salary'] = le.fit_transform(df['salary'])
df.head()

print("X head:")
x = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y = df['left']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
y_pred

accuracy = accuracy_score(y_test, y_pred)
accuracy

print("Data Prediction")
dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])

plt.figure(figsize=(10,10))
plot_tree(dt, filled=True, feature_names=x.columns, class_names=['salary','left'])
plt.show()
```

## Output:

![image](https://github.com/user-attachments/assets/97f123a1-35ad-438d-8498-ca2d5e2f09f5)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
