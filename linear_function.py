import tensorflow as tf
from tensorflow.keras import layers, models
import visualkeras
import numpy as np


model = models.Sequential([
    layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

#2x-1
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(x, y, epochs=500)

model.summary()

print("19: ", model.predict(np.array([10.0]))) 
print("-21: ", model.predict(np.array([-10.0]))) 

weights = model.layers[0].get_weights()

print(f"Weights: {weights[0][0][0]}")
print(f"Bias: {weights[1][0]}")

#visualkeras.layered_view(model, to_file='cnn_model.png',).show()