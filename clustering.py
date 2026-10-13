
# AIML Assignment 2
# Apply Clustering using Unsupervised Learning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Create sample dataset
data = {
    "Study_Hours": [1, 2, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [20, 25, 30, 35, 50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Step 2: Select features
X = df[["Study_Hours", "Marks"]]

# Step 3: Apply K-Means clustering
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X)

# Step 4: Display clustered data
print("\nDataset after Clustering:")
print(df)

# Step 5: Display cluster centers
print("\nCluster Centers:")
print(model.cluster_centers_)

# Step 6: Visualize clusters
for cluster_number in sorted(df["Cluster"].unique()):
    cluster_data = df[df["Cluster"] == cluster_number]
    plt.scatter(
        cluster_data["Study_Hours"],
        cluster_data["Marks"],
        label=f"Cluster {cluster_number}"
    )

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("K-Means Clustering using Unsupervised Learning")
plt.legend()
plt.grid(True)
plt.show()
