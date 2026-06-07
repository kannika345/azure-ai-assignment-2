import matplotlib
matplotlib.use('TkAgg')
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("--- PCA SCRIPT STARTED ---")

# 1. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 3. Visualization
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y, cmap="bwr", edgecolors="k")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection (Iris Dataset)")
plt.savefig("pca.png")   # saves PNG file
plt.show()



