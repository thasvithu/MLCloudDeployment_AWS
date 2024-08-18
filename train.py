# train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv("Admission_Predict.csv")

# Handle missing values
df["GRE Score"] = df["GRE Score"].fillna(df["GRE Score"].mean())
df["TOEFL Score"] = df["TOEFL Score"].fillna(df["TOEFL Score"].mean())
df["University Rating"] = df["University Rating"].fillna(df["University Rating"].mean())

# Drop the Serial No. column
df.drop(columns=["Serial No."], inplace=True)

# Create label and feature columns
y = df["Chance of Admit "]
x = df.drop(columns=["Chance of Admit "])

# Standardize features
scaler = StandardScaler()
x_standardized = scaler.fit_transform(x)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x_standardized, y, test_size=0.20, random_state=100)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Save the model and scaler
pickle.dump(model, open("admission_model.pickle", "wb"))
pickle.dump(scaler, open("scaler.pickle", "wb"))

print("Model and scaler saved successfully.")
