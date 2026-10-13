
# Aim: Fake News Detection using Multimodal Data
# Modalities used: News Text + News Image

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix

# Step 1: Create a folder for sample images
os.makedirs("news_images", exist_ok=True)

# Step 2: Create sample image files
# These are demonstration images, not real news photographs.
def create_sample_image(filename, color, pattern):
    image = Image.new("RGB", (100, 100), color)
    draw = ImageDraw.Draw(image)

    if pattern == "circle":
        draw.ellipse((20, 20, 80, 80), fill="white")
    else:
        draw.rectangle((20, 20, 80, 80), fill="white")

    image.save(filename)

# Step 3: Prepare sample news data
texts = [
    "Government announces new public transport project",
    "Scientists publish results of climate research",
    "City hospital opens a new medical facility",
    "University announces scholarship applications",
    "Weather department issues heavy rain advisory",
    "Local team wins the national sports final",
    "Official report presents economic growth figures",
    "New railway service begins after safety inspection",
    "Shocking miracle cure works instantly for everyone",
    "Celebrity secretly controls the entire country",
    "Forward this message to receive free money today",
    "Unbelievable secret proves all experts are lying",
    "Urgent claim says a magic drink cures every disease",
    "Secret message promises guaranteed lottery winnings",
    "Viral post claims the moon disappeared last night",
    "Anonymous message announces an impossible discovery"
]

labels = [
    "REAL", "REAL", "REAL", "REAL",
    "REAL", "REAL", "REAL", "REAL",
    "FAKE", "FAKE", "FAKE", "FAKE",
    "FAKE", "FAKE", "FAKE", "FAKE"
]

# Create demonstration images for each sample
image_paths = []

for i in range(len(texts)):
    path = f"news_images/news_{i}.png"

    if labels[i] == "REAL":
        create_sample_image(path, "lightblue", "rectangle")
    else:
        create_sample_image(path, "salmon", "circle")

    image_paths.append(path)

data = pd.DataFrame({
    "text": texts,
    "image_path": image_paths,
    "label": labels
})

print("Sample dataset created:")
print(data[["text", "label"]].head())

# Step 4: Extract text features using TF-IDF
text_vectorizer = TfidfVectorizer()
text_features = text_vectorizer.fit_transform(data["text"])

# Step 5: Extract basic image features
def extract_image_features(path):
    image = Image.open(path).convert("RGB").resize((50, 50))
    pixels = np.array(image, dtype=float)

    # Average red, green and blue values
    rgb_means = pixels.mean(axis=(0, 1))

    # Average brightness and pixel variation
    brightness = pixels.mean()
    contrast = pixels.std()

    return [
        rgb_means[0],
        rgb_means[1],
        rgb_means[2],
        brightness,
        contrast
    ]

image_features = np.array([
    extract_image_features(path)
    for path in data["image_path"]
])

# Scale image features
scaler = StandardScaler()
image_features_scaled = scaler.fit_transform(image_features)

# Step 6: Combine text and image features
combined_features = hstack([
    text_features,
    csr_matrix(image_features_scaled)
])

# Step 7: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    combined_features,
    data["label"],
    test_size=0.25,
    random_state=42,
    stratify=data["label"]
)

# Step 8: Train the multimodal classification model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 9: Evaluate the model
predictions = model.predict(X_test)

print("\nMultimodal Fake News Detection Results")
print("--------------------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions), 2))
print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

# Step 10: Predict a sample news item
sample_text = [
    "Urgent viral message promises a guaranteed miracle cure"
]

sample_text_features = text_vectorizer.transform(sample_text)

sample_image_features = np.array([
    extract_image_features("news_images/news_8.png")
])

sample_image_scaled = scaler.transform(sample_image_features)

sample_combined = hstack([
    sample_text_features,
    csr_matrix(sample_image_scaled)
])

prediction = model.predict(sample_combined)[0]
probabilities = model.predict_proba(sample_combined)[0]
confidence = max(probabilities) * 100

print("\nSample News Prediction")
print("----------------------")
print("News:", sample_text[0])
print("Predicted Label:", prediction)
print("Model Confidence:", round(confidence, 2), "%")

# Step 11: Plot test results
plt.figure(figsize=(6, 4))

counts = pd.Series(predictions).value_counts()
counts.plot(kind="bar")

plt.title("Predicted News Categories")
plt.xlabel("Category")
plt.ylabel("Number of Test Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nProgram completed successfully.")
print("Sample images are stored in the news_images folder.")
