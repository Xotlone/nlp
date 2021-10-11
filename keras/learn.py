import os

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

train_x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
train_y = np.array([[0, 1, 1, 0]]).T

model = Sequential([
    Dense(2, 'relu'),
    Dense(1, 'sigmoid')
])

if os.path.isfile('model.h5') and os.path.isfile('weights.hdf5'):
    model = keras.models.load_model('model.h5')
    model.load_weights('weights.hdf5')
else:
    model.compile(
        'adam',
        'binary_crossentropy',
        metrics=['accuracy']
    )

    metrics = model.fit(
        train_x, train_y,
        epochs=5000,
    )

    #model.save('model.h5')
    model.save_weights('weights.hdf5')

    r = list(range(len(metrics.history['loss'])))
    plt.plot(r, metrics.history['loss'], r, metrics.history['accuracy'])
    plt.show()

while True:
    i = np.array([[int(input('1: ')), int(input('2: '))]])
    print(model.predict(i)[0][0])