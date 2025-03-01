import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set the path to the dataset folder
data_path = r"C:\Users\mustapha\Downloads\dsc6232-rwanda-summer2020-hw2"

# Load training features (X) and labels (y)
X_train = pd.read_csv(os.path.join(data_path, "X_train.txt"), delim_whitespace=True, header=None)
y_train = pd.read_csv(os.path.join(data_path, "y_train.txt"), delim_whitespace=True, header=None)

# Load test features
X_test = pd.read_csv(os.path.join(data_path, "X_test_df.csv"))

# Rename target column for clarity
y_train.columns = ['Deforestation_Label']

# Check for missing values
print("Missing values in X_train:", X_train.isnull().sum().sum())
print("Missing values in y_train:", y_train.isnull().sum().sum())
print("Missing values in X_test:", X_test.isnull().sum().sum())

# Fill missing values if necessary
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Split training data into training and validation sets (80% train, 20% validation)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split.values.ravel())  # Convert y_train to a 1D array

# Evaluate the model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Save the predictions to a CSV file
output_df = pd.DataFrame({'Predicted_Deforestation': y_test_pred})
output_df.to_csv(os.path.join(data_path, "deforestation_predictions.csv"), index=False)

print("Predictions saved successfully!")
