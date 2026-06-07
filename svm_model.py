import matplotlib
matplotlib.use('TkAgg')
import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

print("--- SVM SCRIPT STARTED ---")

def load_data(data_dir):
    images = []
    labels = []
    if not os.path.exists(data_dir):
        print(f"ERROR: Folder '{data_dir}' not found!")
        return np.array([]), np.array([])
    
    categories = ['cat', 'dog'] # Make sure these match your folder names exactly
    for category in categories:
        path = os.path.join(data_dir, category)
        print(f"Checking folder: {path}")
        if not os.path.exists(path):
            print(f"  Warning: Subfolder {category} not found.")
            continue
            
        count = 0
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64)) 
                images.append(img.flatten())
                labels.append(category)
                count += 1
        print(f"  Loaded {count} images for {category}")
                
    return np.array(images), np.array(labels)

# Load and process
X_train, y_train = load_data('train')
X_val, y_val = load_data('val')

if len(X_train) == 0:
    print("FATAL ERROR: No training data found. Check your folder paths!")
else:
    print("Training SVM... please wait...")
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    print(f"DONE! SVM Accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%")
    joblib.dump(model, 'my_svm_model.pkl')
    from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()
