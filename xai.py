import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
import matplotlib as mpl
from matplotlib import pyplot as plt

def get_gradcam_plusplus(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.outputs]
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        conv_outputs, preds = grad_model(img_tensor)
        
        if isinstance(preds, list):
            preds = preds[0]
        if isinstance(conv_outputs, list):
            conv_outputs = conv_outputs[0]

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    first_derivative = grads
    second_derivative = grads * grads
    third_derivative = grads * grads * grads

    global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1, 2))
    
    alpha_num = second_derivative
    alpha_denom = 2.0 * second_derivative + third_derivative * tf.reshape(global_sum, (1, 1, 1, -1))
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    
    alphas = alpha_num / alpha_denom

    weights = tf.maximum(first_derivative, 0.0)
    
    alphas_thresholding = tf.where(weights != 0.0, alphas, tf.zeros_like(alphas))
    alpha_normalization_constant = tf.reduce_sum(alphas_thresholding, axis=(1, 2))
    alpha_normalization_constant = tf.reshape(alpha_normalization_constant, (1, 1, 1, -1))
    alpha_normalization_constant = tf.where(alpha_normalization_constant != 0.0, 
                                            alpha_normalization_constant, 
                                            tf.ones_like(alpha_normalization_constant))

    alphas /= alpha_normalization_constant
    
    deep_linearization_weights = tf.reduce_sum(weights * alphas, axis=(1, 2))

    cam = tf.reduce_sum(deep_linearization_weights[:, tf.newaxis, tf.newaxis, :] * conv_outputs, axis=-1)
    
    cam = tf.maximum(cam, 0.0)
    cam = cam / (tf.reduce_max(cam) + 1e-10)

    return tf.squeeze(cam).numpy()

def get_gradcam(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Modell összeállítása a gradiensek és aktivációk kinyeréséhez
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.outputs]
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        conv_outputs, preds = grad_model(img_tensor)
        
        if isinstance(preds, list):
            preds = preds[0]
        if isinstance(conv_outputs, list):
            conv_outputs = conv_outputs[0]

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 2. Gradiensek kiszámítása (első derivált)
    grads = tape.gradient(class_channel, conv_outputs)

    # 3. Grad-CAM lényege: a gradiensek térbeli átlagolása (Global Average Pooling)
    # Az 1-es és 2-es tengely a magasság és a szélesség.
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # 4. A súlyok rávetítése a konvolúciós réteg aktivációira
    # Kibővítjük a dimenziót (batch, 1, 1, channels), hogy össze lehessen szorozni a konv. kimenettel
    pooled_grads = pooled_grads[:, tf.newaxis, tf.newaxis, :]
    
    # Kiszámoljuk a súlyozott összeget a csatornák (utolsó tengely) mentén
    cam = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    
    # 5. ReLU alkalmazása (csak a pozitív, a predikciót segítő értékek kellenek) és normalizálás
    cam = tf.maximum(cam, 0.0)
    cam = cam / (tf.reduce_max(cam) + 1e-10)

    return tf.squeeze(cam).numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    heatmap = tf.expand_dims(heatmap, -1)
    heatmap = tf.image.resize(heatmap, (img.shape[0], img.shape[1]))
    heatmap = tf.squeeze(heatmap).numpy()

    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[np.uint8(heatmap * 255)]

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 1)

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Eredeti kép")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM++ Hőtérkép")
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Rávetítés (Overlay)")
    plt.imshow(superimposed_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()