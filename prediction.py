import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

dataset = pd.read_csv(r'C:\Users\HP\Downloads\archive (7).zip')
print(dataset)

print(dataset.isnull().sum())
print(dataset.describe())
print(dataset.drop(columns = ['id']))
print(dataset.columns)
dataset['diagnosis']=dataset['diagnosis'].map({'M':1, 'B':0})
print(dataset)

correlation =dataset.corr(numeric_only=True)
print(correlation)

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cmap='coolwarm')
plt.title('Correlation between different features')
plt.show()

plt.scatter(dataset['diagnosis'], dataset['radius_mean'])
plt.xlabel('Diagnosis')
plt.ylabel('Radius Mean')
plt.title('Scatter Plot of Diagnosis vs Radius Mean')
plt.show()


plt.scatter(dataset['radius_mean'], dataset['texture_mean'])
plt.xlabel('Radius Mean')
plt.ylabel('Texture Mean')
plt.title('Scatter Plot of Radius Mean vs Texture Mean')
plt.show()

plt.scatter(dataset['perimeter_mean'], dataset['area_mean'])
plt.xlabel('Perimeter Mean')
plt.ylabel('Area Mean') 
plt.title('Scatter Plot of Perimeter Mean vs Area Mean')
plt.show()

average_radius = dataset.groupby('diagnosis')['radius_mean'].mean()
plt.bar(average_radius.index, average_radius.values)
plt.xlabel('Diagnosis')
plt.ylabel('Average Radius Mean')
plt.title('Average Radius Mean by Diagnosis')
plt.show()

plt.scatter(dataset['radius_worst'], dataset['radius_mean'])
plt.xlabel('Radius Worst')
plt.ylabel('Radius Mean')
plt.title('Scatter Plot of Radius Worst vs Radius Mean')
plt.show()

y= dataset['diagnosis']
x= dataset[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'radius_worst']]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
   
joblib.dump(
    model,
    r"C:\Users\HP\breast_cancer_prediction_project.py\breast_cancer_model.pkl"
)   

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

class_report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{class_report}")

r2_score_value = r2_score(y_test, y_pred)
print(f"R2 Score: {r2_score_value}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Predicted Benign', 'Predicted Malignant'], yticklabels=['Actual Benign', 'Actual Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

new_person = [[20.77, 130, 1200, 20, 25]]

prediction = model.predict(new_person)

if prediction[0] == 1:
    print("Prediction: Malignant")
else:
    print("Prediction: Benign")
    
new_person = [[25.63, 30.26, 32.22, 17.23, 10.24]]

prediction = model.predict(new_person)

if prediction[0] == 1:
    print("Prediction: Malignant")
else:
    print("Prediction: Benign")