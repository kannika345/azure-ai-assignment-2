import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("--- RANDOM FOREST SCRIPT STARTED ---")

# 1. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
data = pd.DataFrame(X, columns=iris.feature_names)
data['label'] = y

# 2. Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Predictions
y_pred = model.predict(X)
print(f"Random Forest Accuracy: {accuracy_score(y, y_pred) * 100:.2f}%")

# 4. Save model
joblib.dump(model, "my_random_forest.pkl")
print("Model saved!")

# 5. Confusion Matrix Visualization
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix (Iris Dataset)")
plt.savefig("random_forest.png")   # saves PNG file
plt.show()
