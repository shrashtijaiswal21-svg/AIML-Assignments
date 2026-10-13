
# Aim: To implement Support Vector Machine (SVM) for Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Create sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [20, 25, 35, 40, 50, 60, 65, 75, 85, 95]
}

df = pd.DataFrame(data)

# Step 2: Separate input and output
X = df[["Study_Hours"]]
y = df["Marks"]

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create and train the SVM regression model
model = SVR(kernel="rbf", C=100, epsilon=1.0)
model.fit(X_train_scaled, y_train)

# Step 6: Predict marks
y_pred = model.predict(X_test_scaled)

# Step 7: Display evaluation metrics
print("SVM Regression Results")
print("----------------------")
print("Actual Marks:", list(y_test))
print("Predicted Marks:", y_pred.round(2))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

# Step 8: Predict marks for a student studying 7 hours
new_student = pd.DataFrame({"Study_Hours": [7]})
new_student_scaled = scaler.transform(new_student)
predicted_marks = model.predict(new_student_scaled)

print("\nPredicted marks for 7 study hours:",
      round(predicted_marks[0], 2))

# Step 9: Plot actual data and SVM regression curve
X_plot = pd.DataFrame({
    "Study_Hours": [i / 10 for i in range(10, 101)]
})
X_plot_scaled = scaler.transform(X_plot)
y_plot = model.predict(X_plot_scaled)

plt.scatter(
    df["Study_Hours"], df["Marks"],
    color="blue", label="Actual Data"
)
plt.plot(
    X_plot["Study_Hours"], y_plot,
    color="red", label="SVM Regression Curve"
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("SVM Regression")
plt.legend()
plt.grid(True)
plt.show()
