import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Confusion matrices
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

# Sensitivity & Specificity function
def sensitivity_specificity(cm):
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity, specificity

# Training metrics
train_acc = accuracy_score(y_train, y_train_pred)
train_sens, train_spec = sensitivity_specificity(cm_train)
train_f1 = f1_score(y_train, y_train_pred)

# Testing metrics
test_acc = accuracy_score(y_test, y_test_pred)
test_sens, test_spec = sensitivity_specificity(cm_test)
test_f1 = f1_score(y_test, y_test_pred)

# Print results
print("======= TRAINING METRICS =======")
print("Accuracy:", train_acc)
print("Sensitivity:", train_sens)
print("Specificity:", train_spec)
print("F1 Score:", train_f1)

print("\n======= TESTING METRICS =======")
print("Accuracy:", test_acc)
print("Sensitivity:", test_sens)
print("Specificity:", test_spec)
print("F1 Score:", test_f1)
