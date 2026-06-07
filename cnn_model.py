import tensorflow as tf
from tensorflow.keras import layers, models

# 1. SETUP DATA PIPELINE
# This looks at your folders and automatically labels the images
train_ds = tf.keras.utils.image_dataset_from_directory(
    'train',
    image_size=(150, 150),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'val',
    image_size=(150, 150),
    batch_size=32
)

# 2. DEFINE THE CNN ARCHITECTURE
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(150, 150, 3)), # Normalizes pixel values
    layers.Conv2D(32, (3, 3), activation='relu'),       # Extracts features
    layers.MaxPooling2D((2, 2)),                        # Reduces data size
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),                                   # Converts to 1D vector
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')               # Change '2' to your number of folders
])

# 3. COMPILE
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. TRAIN
print("Starting training...")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 5. SAVE
model.save('my_cnn_model.h5')
print("Model saved!")
import matplotlib.pyplot as plt

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Training Accuracy')
plt.legend()
plt.show()
