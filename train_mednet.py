import tensorflow as tf
from medmnist import BloodMNIST
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

size = 128
train_dataset = BloodMNIST(split='train', download=True, size=size)
test_dataset = BloodMNIST(split='test', download=True, size=size)
val_dataset = BloodMNIST(split='val', download=True, size=size)

x_train = train_dataset.imgs.astype('float32') / 255.0
y_train = np.squeeze(train_dataset.labels)

x_val = val_dataset.imgs.astype('float32') / 255.0
y_val = np.squeeze(val_dataset.labels)

# Modell betöltése (architektúra, súlyok és optimizer állapot is jön vele)
model = tf.keras.models.load_model('mednet_modell_128_11.keras')

datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.1
)

# Továbbtanítás újabb 1 epochos lépésben
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=16), 
    epochs=20,
    validation_data=(x_val, y_val),
)

# Felülírjuk az előző mentést a frissített súlyokkal
model.save('mednet_modell_128_31.keras')

#748/748 [==============================] - 314s 417ms/step - loss: 0.4583 - accuracy: 0.8424 - val_loss: 0.2403 - val_accuracy: 0.9130
#748/748 [==============================] - 2777s 4s/step - loss: 0.2244 - accuracy: 0.9226 - val_loss: 0.2225 - val_accuracy: 0.9211
'''
748/748 [==============================] - 312s 415ms/step - loss: 0.2271 - accuracy: 0.9210 - val_loss: 0.4107 - val_accuracy: 0.8686
Epoch 2/10
748/748 [==============================] - 310s 414ms/step - loss: 0.1878 - accuracy: 0.9337 - val_loss: 0.2846 - val_accuracy: 0.8908
Epoch 3/10
748/748 [==============================] - 310s 414ms/step - loss: 0.1531 - accuracy: 0.9456 - val_loss: 0.2347 - val_accuracy: 0.9223
Epoch 4/10
748/748 [==============================] - 309s 413ms/step - loss: 0.1347 - accuracy: 0.9516 - val_loss: 0.8855 - val_accuracy: 0.7687
Epoch 5/10
748/748 [==============================] - 309s 413ms/step - loss: 0.1273 - accuracy: 0.9574 - val_loss: 0.1463 - val_accuracy: 0.9463
Epoch 6/10
748/748 [==============================] - 309s 413ms/step - loss: 0.1283 - accuracy: 0.9552 - val_loss: 0.3449 - val_accuracy: 0.8995
Epoch 7/10
748/748 [==============================] - 309s 413ms/step - loss: 0.1125 - accuracy: 0.9619 - val_loss: 0.1697 - val_accuracy: 0.9387
Epoch 8/10
748/748 [==============================] - 309s 413ms/step - loss: 0.1031 - accuracy: 0.9638 - val_loss: 0.1603 - val_accuracy: 0.9445
Epoch 9/10
748/748 [==============================] - 308s 412ms/step - loss: 0.1015 - accuracy: 0.9655 - val_loss: 0.4287 - val_accuracy: 0.8732
Epoch 10/10
748/748 [==============================] - 306s 410ms/step - loss: 0.1001 - accuracy: 0.9669 - val_loss: 0.1930 - val_accuracy: 0.9282
748/748 [==============================] - 305s 406ms/step - loss: 0.0956 - accuracy: 0.9687 - val_loss: 0.2216 - val_accuracy: 0.9246
Epoch 2/20
748/748 [==============================] - 306s 409ms/step - loss: 0.0885 - accuracy: 0.9679 - val_loss: 0.1081 - val_accuracy: 0.9603
Epoch 3/20
748/748 [==============================] - 306s 409ms/step - loss: 0.0722 - accuracy: 0.9732 - val_loss: 0.1421 - val_accuracy: 0.9585
Epoch 4/20
748/748 [==============================] - 306s 409ms/step - loss: 0.0799 - accuracy: 0.9717 - val_loss: 0.1690 - val_accuracy: 0.9433
Epoch 5/20
748/748 [==============================] - 306s 409ms/step - loss: 0.0897 - accuracy: 0.9681 - val_loss: 0.0615 - val_accuracy: 0.9819
Epoch 6/20
748/748 [==============================] - 307s 410ms/step - loss: 0.0755 - accuracy: 0.9762 - val_loss: 0.0771 - val_accuracy: 0.9784
Epoch 7/20
748/748 [==============================] - 306s 409ms/step - loss: 0.0773 - accuracy: 0.9742 - val_loss: 0.0867 - val_accuracy: 0.9714
Epoch 8/20
748/748 [==============================] - 306s 409ms/step - loss: 0.0721 - accuracy: 0.9749 - val_loss: 0.1427 - val_accuracy: 0.9597
Epoch 9/20
748/748 [==============================] - 305s 408ms/step - loss: 0.0639 - accuracy: 0.9775 - val_loss: 0.0539 - val_accuracy: 0.9831
Epoch 10/20
748/748 [==============================] - 306s 409ms/step - loss: 0.0713 - accuracy: 0.9753 - val_loss: 0.0771 - val_accuracy: 0.9749
Epoch 11/20
748/748 [==============================] - 306s 409ms/step - loss: 0.0621 - accuracy: 0.9783 - val_loss: 0.1633 - val_accuracy: 0.9393
Epoch 12/20
748/748 [==============================] - 306s 409ms/step - loss: 0.0635 - accuracy: 0.9782 - val_loss: 0.0610 - val_accuracy: 0.9825
Epoch 13/20
748/748 [==============================] - 308s 411ms/step - loss: 0.0633 - accuracy: 0.9787 - val_loss: 0.1026 - val_accuracy: 0.9650
Epoch 14/20
748/748 [==============================] - 308s 412ms/step - loss: 0.0606 - accuracy: 0.9793 - val_loss: 0.0503 - val_accuracy: 0.9825
Epoch 15/20
748/748 [==============================] - 309s 412ms/step - loss: 0.0570 - accuracy: 0.9806 - val_loss: 0.1659 - val_accuracy: 0.9369
Epoch 16/20
748/748 [==============================] - 312s 417ms/step - loss: 0.0621 - accuracy: 0.9774 - val_loss: 0.0705 - val_accuracy: 0.9749
Epoch 17/20
748/748 [==============================] - 312s 418ms/step - loss: 0.0609 - accuracy: 0.9790 - val_loss: 0.0757 - val_accuracy: 0.9772
Epoch 18/20
748/748 [==============================] - 312s 418ms/step - loss: 0.0570 - accuracy: 0.9800 - val_loss: 0.2095 - val_accuracy: 0.9229
Epoch 19/20
748/748 [==============================] - 311s 415ms/step - loss: 0.0488 - accuracy: 0.9820 - val_loss: 0.0825 - val_accuracy: 0.9650
Epoch 20/20
748/748 [==============================] - 313s 418ms/step - loss: 0.0567 - accuracy: 0.9789 - val_loss: 0.0866 - val_accuracy: 0.9685
'''