import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

print("--- DECISION TREE SCRIPT STARTED ---")

# 1. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
data = pd.DataFrame(X, columns=iris.feature_names)
data['label'] = y

# 2. Train Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X, y)

# 3. Predictions
y_pred = model.predict(X)
print(f"Decision Tree Accuracy: {accuracy_score(y, y_pred) * 100:.2f}%")

# 4. Save model
joblib.dump(model, "my_decision_tree.pkl")
print("Model saved!")

# 5. Visualization
plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Visualization (Iris Dataset)")
plt.savefig("decision_tree.png")   # saves the graph as PNG
plt.show()
