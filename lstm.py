import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("--- LSTM SCRIPT STARTED ---")

# 1. Create synthetic dataset
timesteps = 10   # length of each sequence
features = 1     # number of features per timestep
X_train = np.random.rand(100, timesteps, features)
y_train = np.random.rand(100, 1)
X_val = np.random.rand(20, timesteps, features)
y_val = np.random.rand(20, 1)

# 2. Build LSTM model
model = models.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(timesteps, features)),
    layers.LSTM(50),
    layers.Dense(1)
])

# 3. Compile and train
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# 4. Save model
model.save("my_lstm_model.h5")
print("Model saved!")

# 5. Plot training history
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training Loss")
plt.legend()
plt.savefig("lstm_training.png")   # saves PNG file
plt.show()
