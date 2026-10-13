
# AIML Assignment 4
# Apply Naive Bayes Classification Algorithm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create sample dataset
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

# Step 2: Separate input features and target
X = df[["Study_Hours", "Attendance"]]
y = df["Result"]

# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Step 4: Create and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 5: Predict test results
y_pred = model.predict(X_test)

print("\nActual Results:")
print(list(y_test))

print("\nPredicted Results:")
print(list(y_pred))

# Step 6: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 7: Predict result for a new student
new_student = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [75]
})

prediction = model.predict(new_student)

print("New Student Study Hours: 6")
print("New Student Attendance: 75%")
print("Predicted Student Result:", prediction[0])

# Step 8: Visualize actual student results

import matplotlib.pyplot as plt

# Plot Fail students
fail_data = df[df["Result"] == "Fail"]
plt.scatter(
    fail_data["Study_Hours"],
    fail_data["Attendance"],
    color="red",
    label="Fail"
)

# Plot Pass students
pass_data = df[df["Result"] == "Pass"]
plt.scatter(
    pass_data["Study_Hours"],
    pass_data["Attendance"],
    color="green",
    label="Pass"
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance (%)")
plt.title("Naive Bayes Classification of Student Results")
plt.legend()
plt.grid(True)
plt.show()
