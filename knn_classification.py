
# AIML Assignment 5
# Apply KNN Algorithm for Classification

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Create dataset
data = {
    "Study_Hours": [
        1, 2, 2, 3, 3, 4, 4, 5,
        5, 6, 6, 7, 7, 8, 8, 9
    ],
    "Attendance": [
        40, 45, 50, 55, 60, 65, 68, 70,
        72, 75, 78, 80, 82, 85, 90, 95
    ],
    "Result": [
        "Fail", "Fail", "Fail", "Fail",
        "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Step 2: Separate features and target
X = df[["Study_Hours", "Attendance"]]
y = df["Result"]

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Step 4: Standardize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create and train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# Step 6: Predict test results
y_pred = model.predict(X_test_scaled)

print("\nActual Results:")
print(list(y_test))

print("\nPredicted Results:")
print(list(y_pred))

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["Fail", "Pass"]))

# Step 8: Predict result for a new student
new_student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [75]
})

new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)

print("\nNew Student Study Hours: 6")
print("New Student Attendance: 75%")
print("Predicted Result:", prediction[0])

# Step 9: Plot student data
for result, color in [("Fail", "red"), ("Pass", "green")]:
    subset = df[df["Result"] == result]
    plt.scatter(
        subset["Study_Hours"],
        subset["Attendance"],
        color=color,
        label=result
    )

# Mark the new student
plt.scatter(
    6, 75,
    color="blue",
    marker="*",
    s=200,
    label="New Student"
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance (%)")
plt.title("KNN Classification of Student Results")
plt.legend()
plt.grid(True)
plt.show()
