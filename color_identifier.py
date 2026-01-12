from typing import List
import tensorflow as tf
from tensorflow.keras import layers, models
import visualkeras
import numpy as np
from image_utils import create_image, visualize_image, red_rbg, green_rbg, blue_rbg

data_size = 1000

green_images = [create_image(4, 'green') for _ in range(data_size)]
red_images = [create_image(4, 'red') for _ in range(data_size)]
blue_images = [create_image(4, 'blue') for _ in range(data_size)]

red_rgbs = [red_rbg for _ in range(data_size)]
green_rgbs = [green_rbg for _ in range(data_size)]
blue_rgbs = [blue_rbg for _ in range(data_size)]

#visualize_image(red_image)

model = models.Sequential([
    layers.Input(shape=(4, 4, 3)), 
    layers.Flatten(), 
    layers.Dense(units=3, activation='relu'),
    layers.Dense(3, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.00001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

#red and green images
x = np.array(red_images +
             green_images +
             blue_images,
             dtype=np.uint8)
y = np.array( [[1, 0, 0] for _ in range(data_size)] +
              [[0, 1, 0] for _ in range(data_size)] +
              [[0, 0, 1] for _ in range(data_size)])

model.fit(x, y, epochs=1000)

red_image = create_image(4, 'red')
green_image = create_image(4, 'green')
blue_image = create_image(4, 'blue')

print("1 - green: ", model.predict(np.array([green_image]))) 
print("0 - red: ", model.predict(np.array([red_image])))
print("0.5 - blue: ", model.predict(np.array([blue_image])))

    