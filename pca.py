
# AIML Assignment 3
# Aim: Apply Dimensionality Reduction using PCA

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Create sample dataset
data = {
    "Maths": [80, 70, 90, 60, 85, 75, 95, 65],
    "Science": [85, 75, 95, 65, 80, 70, 90, 60],
    "English": [70, 65, 80, 55, 75, 68, 85, 60]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Step 2: Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 3: Apply PCA and reduce to 2 components
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Step 4: Display reduced dataset
pca_df = pd.DataFrame(
    reduced_data,
    columns=["Principal Component 1", "Principal Component 2"]
)

print("\nDataset After PCA:")
print(pca_df.round(3))

# Step 5: Display explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_.round(3))

print(
    "\nTotal Variance Retained:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

# Step 6: Visualize the reduced data
plt.scatter(
    pca_df["Principal Component 1"],
    pca_df["Principal Component 2"]
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Dimensionality Reduction using PCA")
plt.grid(True)
plt.show()
