import tensorflow as tf
import numpy as np
import cv2
import joblib
import os
from tensorflow.keras.preprocessing import image

print("--- STARTING FINAL COMPARISON ---")

# 1. AUTO-FIND AN IMAGE
dog_folder = 'val/dog'
available_files = [f for f in os.listdir(dog_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
test_img = os.path.join(dog_folder, available_files[0])
print(f"Testing with image: {test_img}")

# 2. LOAD MODELS
cnn_model = tf.keras.models.load_model('my_cnn_model.h5')
svm_model = joblib.load('my_svm_model.pkl')

# 3. PREPARE FOR CNN (150x150)
img_for_cnn = image.load_img(test_img, target_size=(150, 150))
x = image.img_to_array(img_for_cnn) / 255.0
x = np.expand_dims(x, axis=0)

# 4. PREPARE FOR SVM (64x64)
raw_img = cv2.imread(test_img)
img_svm = cv2.resize(raw_img, (64, 64)).flatten().reshape(1, -1) / 255.0

# 5. GET PREDICTIONS
cnn_pred = cnn_model.predict(x)
cnn_result = "Dog" if cnn_pred[0][0] > 0.5 else "Cat"

svm_result = svm_model.predict(img_svm)[0]

# 6. PRINT RESULTS
print("\n" + "="*30)
print(f"CNN SAYS: {cnn_result} ({cnn_pred[0][0]:.2f})")
print(f"SVM SAYS: {svm_result}")
print("="*30)