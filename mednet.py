import tensorflow as tf
from keras.layers import (Input, Conv2D, SeparableConv2D, BatchNormalization, 
                                     ReLU, GlobalAveragePooling2D, GlobalMaxPooling2D, 
                                     Reshape, Dense, Multiply, Concatenate, Add, Dropout)
from keras.models import Model
import keras
from medmnist import BloodMNIST
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

size = 128

def cbam_module(inputs, reduction_ratio=8):
    channels = inputs.shape[-1]

    avg_pool = GlobalAveragePooling2D()(inputs)
    max_pool = GlobalMaxPooling2D()(inputs)

    mlp_1 = Dense(channels // reduction_ratio, activation='relu', use_bias=False)
    mlp_2 = Dense(channels, activation=None, use_bias=False)

    avg_out = mlp_2(mlp_1(avg_pool))
    max_out = mlp_2(mlp_1(max_pool))

    channel_attention = Add()([avg_out, max_out])
    channel_attention = tf.keras.activations.sigmoid(channel_attention)
    channel_attention = Reshape((1, 1, channels))(channel_attention)

    cbam_feature = Multiply()([inputs, channel_attention])

    avg_pool_spatial = tf.reduce_mean(cbam_feature, axis=-1, keepdims=True)
    max_pool_spatial = tf.reduce_max(cbam_feature, axis=-1, keepdims=True)
    spatial_concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])

    spatial_attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', use_bias=False)(spatial_concat)

    return Multiply()([cbam_feature, spatial_attention])

def residual_dscbam_block(inputs, filters, stride=1):
    """A hálózat magját alkotó ResidualDSCBAMBlock."""
    # 1. Depthwise Separable Convolution
    x = SeparableConv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 2. Depthwise Separable Convolution
    x = SeparableConv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 3. CBAM figyelem modul
    x = cbam_module(x)

    # 4. Residual (Shortcut) kapcsolat
    shortcut = inputs
    
    # Ha a térbeli méret (stride > 1) csökken, vagy a csatornák száma eltér, 
    # 1x1 konvolúcióval hozzuk közös dimenzióba, ahogy a leírás is kéri.
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(inputs)
        shortcut = BatchNormalization()(shortcut) # Kerasban a BN ajánlott a 1x1 conv után is

    # Hozzáadjuk a shortcutot (reziduális kapcsolat)
    x = Add()([x, shortcut])
    return x

def build_mednet(input_shape=(size, size, 3), num_classes=2):
    """A MedNet modell felépítése az 5 fázis alapján."""
    inputs = Input(shape=input_shape)

    # Stage 1: 64 filter, stride 1
    x = residual_dscbam_block(inputs, filters=64, stride=1)

    # Stage 2: 128 filter, stride 2
    x = residual_dscbam_block(x, filters=128, stride=2)

    # Stage 3: 256 filter, stride 2
    x = residual_dscbam_block(x, filters=256, stride=2)

    # Stage 4: 512 filter, stride 2
    x = residual_dscbam_block(x, filters=512, stride=2)

    # Stage 5: 1024 filter, stride 2
    x = residual_dscbam_block(x, filters=1024, stride=2)

    x = keras.layers.Activation('linear', name='gradcam_target_layer')(x)

    # Classifier Head (Osztályozó modul)
    x = GlobalAveragePooling2D()(x) # "Adaptive average pooling" Keras megfelelője
    x = Dropout(0.4)(x)

    # Első FC (Linear) réteg
    # Megjegyzés: A tanulmány szövege ellentmondásos (fent 256-ot, lent 512-t ír). 
    # Itt a 256-ot használtam az első bekezdés specifikációja alapján.
    x = Dense(256, activation='relu')(x)

    # Végső klasszifikációs réteg
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name="MedNet")


model = build_mednet(input_shape=(size, size, 3), num_classes=8)



train_dataset = BloodMNIST(split='train', download=True, size=size)
test_dataset = BloodMNIST(split='test', download=True, size=size)
val_dataset = BloodMNIST(split='val', download=True, size=size)

x_train = train_dataset.imgs.astype('float32') / 255.0
y_train = np.squeeze(train_dataset.labels)

x_val = val_dataset.imgs.astype('float32') / 255.0
y_val = np.squeeze(val_dataset.labels)

x_test = test_dataset.imgs.astype('float32') / 255.0
y_test = np.squeeze(test_dataset.labels)


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate = 0.0003),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.1
)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=16), 
    epochs=1,
    validation_data=(x_val, y_val),
)

model.save('mednet_modell_112_1.keras')