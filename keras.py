import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)

import numpy as np
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, size=(1000,))

model = keras.Sequential([
    layers.Dense(32, input_shape=(10,)),
    layers.Dense(2)
])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(X, y, batch_size=1)