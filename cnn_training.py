import keras
from medmnist import BloodMNIST
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
import matplotlib as mpl
from keras.preprocessing.image import ImageDataGenerator

size = 224
train_dataset = BloodMNIST(split='train', download=True, size=size)
test_dataset = BloodMNIST(split='test', download=True, size=size)
val_dataset = BloodMNIST(split='val', download=True, size=size)

x_train = train_dataset.imgs.astype('float32') / 255.0
y_train = np.squeeze(train_dataset.labels)

x_val = val_dataset.imgs.astype('float32') / 255.0
y_val = np.squeeze(val_dataset.labels)

x_test = test_dataset.imgs.astype('float32') / 255.0
y_test = np.squeeze(test_dataset.labels)

model = keras.Sequential([
    keras.layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(size, size, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('gelu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('gelu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('gelu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('gelu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(256, (3, 3), padding='same', name='last_conv', kernel_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('gelu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='gelu', kernel_initializer='he_normal'),
    keras.layers.Dense(8, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=0.00001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.1
)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=16), 
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]
)

model.evaluate(x_test, y_test)

model.save('cnn.keras')