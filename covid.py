

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st



# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('Covid.csv')

# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.tail()

# number of rows and columns in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target Variable
heart_data['Infected'].value_counts()



X = heart_data.drop(columns='Infected', axis=1)
Y = heart_data['Infected']

print(X)

print(Y)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)



model = LogisticRegression()

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)



# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)
a=st.number_input("Age")
b=st.number_input("Fever")
c=st.number_input("Cough")
d=st.number_input("Breathing Issues")

input_data = (a,b,c,d)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
if st.button("Predict")
    if (prediction[0]== 0):
        st.write('The Person does not have a Covid')
    else:
        st.write('The Person has Covid')
