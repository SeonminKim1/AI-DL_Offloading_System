from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mp
from time import time
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import cv2

batch_size = 100

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train=X_train[:2000]
y_train=y_train[:2000]
X_test=X_test[:2000]
y_test=y_test[:2000]

##### CONVERT DIMENSIONS FOR CONV2D
X_train = (X_train[:,:,:,np.newaxis]/255)
X_test = (X_test[:,:,:,np.newaxis]/255)

##### ON TRAINING DATA
y_train_ohe = np.zeros(((y_train.size, y_train.max()+1)))
y_train_ohe[np.arange(y_train.size),y_train] = 1

##### ON TESTING DATA
y_test_ohe = np.zeros(((y_test.size, y_test.max()+1)))
y_test_ohe[np.arange(y_test.size),y_test] = 1

##### GET INPUT SHAPE
(_, xdim, ydim, zdim) = X_train.shape

##### ADD LAYERS
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(xdim,ydim,zdim), dtype=tf.float32))
model.add(keras.layers.Conv2D(16, (3, 3)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

##### COMPILE MODEL AND PRINT SUMMARY
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

##### PRINT SUMMARY
print(model.summary())

################################################################################
# %% INIT CALLBACKS
################################################################################

tensorboard = TensorBoard(log_dir='logs', update_freq='epoch', batch_size=batch_size)
eearlystopping = EarlyStopping(monitor='val_loss', patience=3)

# %% RUN MODEL
history = model.fit(
    x=X_train,
    y=y_train_ohe,
    epochs=1,
    verbose=1,
    validation_data=(X_test, y_test_ohe),
    use_multiprocessing=True,
    batch_size=batch_size,
    callbacks=[tensorboard, eearlystopping]
)

model.save('keras_model.h5')
