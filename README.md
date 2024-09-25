# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, clean it, and separate features (X) from the target variable (y).
2. Split the data into training and testing sets (e.g., 80% train, 20% test).
3. Create and fit a Logistic Regression model using the training data (X_train, y_train).
4. Make predictions on the test set (X_test) and evaluate performance with accuracy, confusion matrix, and classification report.

## Program:
```py
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

uploaded = files.upload()
data = pd.read_csv(next(iter(uploaded)))

data = data.drop(['sl_no', 'salary'], axis=1)

label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])  # Male = 1, Female = 0
data['ssc_b'] = label_encoder.fit_transform(data['ssc_b'])    # Central = 1, Others = 0
data['hsc_b'] = label_encoder.fit_transform(data['hsc_b'])
data['hsc_s'] = label_encoder.fit_transform(data['hsc_s'])    # Commerce/Science/Arts
data['degree_t'] = label_encoder.fit_transform(data['degree_t'])
data['workex'] = label_encoder.fit_transform(data['workex'])  # Yes = 1, No = 0
data['specialisation'] = label_encoder.fit_transform(data['specialisation'])
data['status'] = label_encoder.fit_transform(data['status'])  # Placed = 1, Not Placed = 0

X = data.drop('status', axis=1)  # Features
y = data['status']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


## Result:
![Screenshot 2024-09-25 091213](https://github.com/user-attachments/assets/40bb3de2-4718-434a-bb9c-324115071c7a)
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
