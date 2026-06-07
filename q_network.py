import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("--- Q-NETWORK SCRIPT STARTED ---")

# 1. Define environment basics (toy example)
state_size = 4    # number of features in state
action_size = 2   # number of possible actions

# 2. Build Q-Network model
model = models.Sequential([
    layers.Dense(24, activation='relu', input_shape=(state_size,)),
    layers.Dense(24, activation='relu'),
    layers.Dense(action_size, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
print(model.summary())

# 3. Create synthetic training data
# States: random values, Actions: one-hot, Rewards: random
X_train = np.random.rand(100, state_size)
y_train = np.random.rand(100, action_size)

# 4. Train Q-Network
history = model.fit(X_train, y_train, epochs=10, verbose=1)

# 5. Save model
model.save("my_q_network.h5")
print("Q-Network model saved!")

# 6. Plot training loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Q-Network Training Loss")
plt.legend()
plt.savefig("q_network_training.png")   # saves PNG file
plt.show()
