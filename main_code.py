import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Convert smoking history into numeric values
df["smoking_history_num"] = df["smoking_history"].map({
    "No Info": -1,
    "never": 0,
    "former": 1,
    "current": 2,
    "not current": 3,
    "ever": 4
})

# Remove old smoking_history column
df.drop("smoking_history", axis=1, inplace=True)

# Features and target
X = df.drop(["diabetes", "gender"], axis=1, errors="ignore")
Y = df["diabetes"]

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1
)

# Train model
Model = DecisionTreeClassifier()
Model.fit(X_train, Y_train)

# Accuracy check
Y_test_pred = Model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

print(f"Model Accuracy: {test_accuracy * 100:.2f}%")

# Save model
with open("MODEL.pkl", "wb") as file:
    pickle.dump(Model, file)

print("Model saved successfully as MODEL.pkl")