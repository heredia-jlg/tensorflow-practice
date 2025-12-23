import tensorflow as tf
from tensorflow.keras import layers, models
import visualkeras


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#build the model to see the summary
model.build(input_shape=(None, 64, 64, 3))
model.summary()


visualkeras.layered_view(model, to_file='cnn_model.png',).show()