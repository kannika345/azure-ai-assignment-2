import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv("dataset.csv", encoding='latin1')

# Keep numeric only
data = data.select_dtypes(include=[np.number])

# Split
X = data.iloc[:200, :-1].values
y = data.iloc[:200, -1].values

# Convert to binary (same as before)
median = np.median(y)
y = (y > median).astype(int)

# Normalize
X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

print("Data ready")

# KNN Class
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]

            prediction = max(set(k_labels), key=k_labels.count)
            predictions.append(prediction)

        return [int(i) for i in predictions]


# Train
model = KNN(k=3)
model.fit(X, y)

# Predict
predictions = model.predict(X)

print("Predictions:", predictions[:10])

# Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y, predictions))