
# Aim: To apply regularization to avoid overfitting

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# Step 1: Create sample dataset
np.random.seed(42)

X = np.linspace(0, 10, 30).reshape(-1, 1)
y = 2 * X.ravel() + 5 + np.random.randn(30) * 3

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 3: Create models
# Polynomial features make the model more complex.
# Ridge and Lasso apply regularization.

models = {
    "Linear Regression": make_pipeline(
        PolynomialFeatures(degree=10),
        StandardScaler(),
        LinearRegression()
    ),

    "Ridge Regression": make_pipeline(
        PolynomialFeatures(degree=10),
        StandardScaler(),
        Ridge(alpha=10)
    ),

    "Lasso Regression": make_pipeline(
        PolynomialFeatures(degree=10),
        StandardScaler(),
        Lasso(alpha=0.1, max_iter=100000)
    )
}

# Step 4: Train models and calculate errors
for name, model in models.items():
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_error = mean_squared_error(y_train, train_predictions)
    test_error = mean_squared_error(y_test, test_predictions)

    print("\n", name)
    print("Training MSE:", round(train_error, 2))
    print("Testing MSE:", round(test_error, 2))

# Step 5: Plot model predictions
X_plot = np.linspace(0, 10, 300).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Actual Data")

for name, model in models.items():
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, label=name)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Regularization to Avoid Overfitting")
plt.legend()
plt.grid(True)
plt.show()
