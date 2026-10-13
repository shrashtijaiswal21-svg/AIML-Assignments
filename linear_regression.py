
# AIML Assignment 6
# Implement Linear Regression Algorithm

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [20, 25, 35, 40, 50, 60, 65, 75, 85, 95]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Step 2: Separate input and output
X = df[["Study_Hours"]]
y = df["Marks"]

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict marks
y_pred = model.predict(X_test)

print("\nActual Marks:")
print(list(y_test))

print("\nPredicted Marks:")
print(y_pred.round(2))

# Step 6: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nRegression Coefficient:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 3))

# Step 7: Predict marks for a new student
new_student = pd.DataFrame({"Study_Hours": [7]})
predicted_marks = model.predict(new_student)

print("\nStudy Hours: 7")
print("Predicted Marks:", round(predicted_marks[0], 2))

# Step 8: Plot the regression line
plt.scatter(
    df["Study_Hours"],
    df["Marks"],
    color="blue",
    label="Actual Data"
)

plt.plot(
    df["Study_Hours"],
    model.predict(X),
    color="red",
    label="Regression Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression: Study Hours vs Marks")
plt.legend()
plt.grid(True)
plt.show()
