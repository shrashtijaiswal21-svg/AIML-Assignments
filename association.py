
# AIML Assignment
# Aim: Identify the association between
# independent and dependent variables

import pandas as pd
import matplotlib.pyplot as plt

# Create dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [35, 40, 50, 55, 65, 70, 80, 85]
}

df = pd.DataFrame(data)

# Display dataset
print("STUDENT DATASET")
print(df)

# Calculate correlation
correlation = df["Study_Hours"].corr(df["Marks"])

print("\nIndependent Variable: Study_Hours")
print("Dependent Variable: Marks")
print("Correlation Coefficient:", round(correlation, 3))

# Identify association
if correlation > 0:
    print("Association: Positive")
elif correlation < 0:
    print("Association: Negative")
else:
    print("Association: No linear correlation")

# Draw graph
plt.scatter(df["Study_Hours"], df["Marks"])
plt.xlabel("Study Hours (Independent Variable)")
plt.ylabel("Marks (Dependent Variable)")
plt.title("Association Between Study Hours and Marks")
plt.grid(True)
plt.show()
