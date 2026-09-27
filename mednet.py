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
EPOCHS = 70
LR_MAX = 3e-4          # Kezdeti learning rate
LR_MIN = 1e-6          # η_min
T_MAX = 20             # T_max (epochban megadva)
WEIGHT_DECAY = 1e-4

class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        """
        Focal Loss inicializálása.
        :param gamma: A könnyű példák súlyának csökkentését szabályozza (általában 2.0).
        :param alpha: Osztály-egyensúlyozó tényező (általában 0.25).
        """
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # 1. Numerikus stabilitás: megakadályozzuk a log(0) kialakulását
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # 2. Sparse címkék (pl. [2, 0, 1]) átalakítása One-hot formátumba (pl. [[0,0,1], [1,0,0], [0,1,0]])
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32) # Biztosítjuk a megfelelő dimenziót
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
        
        # 3. Keresztentrópia (Cross Entropy) kiszámítása
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        
        # 4. Focal Loss modulációs tényező: alpha * (1 - p_t)^gamma
        weight = self.alpha * tf.math.pow(1.0 - y_pred, self.gamma)
        
        # 5. Végső loss kiszámítása
        focal_loss = weight * cross_entropy
        
        # Szummázás az osztályok mentén (mivel csak a helyes osztálynál lesz 0-tól eltérő érték)
        return tf.reduce_sum(focal_loss, axis=-1)

def cosine_annealing(epoch, lr):
    t_cur = epoch % (2 * T_MAX)
    
    if t_cur > T_MAX:
        t_cur = (2 * T_MAX) - t_cur
        
    new_lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * t_cur / T_MAX))
    return new_lr

optimizer = tf.keras.optimizers.experimental.AdamW(
    learning_rate=LR_MAX,
    weight_decay=WEIGHT_DECAY,
    jit_compile=False
)

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
    x = SeparableConv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = SeparableConv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = cbam_module(x)

    shortcut = inputs
    
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(inputs)
        shortcut = BatchNormalization()(shortcut) 

    x = Add()([x, shortcut])
    return x

def build_mednet(input_shape=(size, size, 3), num_classes=2):
    inputs = Input(shape=input_shape)

    x = residual_dscbam_block(inputs, filters=64, stride=1)
    x = residual_dscbam_block(x, filters=128, stride=2)
    x = residual_dscbam_block(x, filters=256, stride=2)
    x = residual_dscbam_block(x, filters=512, stride=2)
    x = residual_dscbam_block(x, filters=1024, stride=2)

    x = keras.layers.Activation('linear', name='gradcam_target_layer')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation='relu')(x)
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
    optimizer=optimizer,
    loss=SparseFocalLoss(gamma=2.0, alpha=0.25),
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