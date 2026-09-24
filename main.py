from xai import get_gradcam_plusplus, display_gradcam
import numpy as np
from tensorflow import keras
from medmnist import BloodMNIST
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf

size = 128

model = tf.keras.models.load_model('mednet_modell_128_31.keras')
test_dataset = BloodMNIST(split='test', download=True, size=size)

x_test = test_dataset.imgs.astype('float32') / 255.0
y_test = np.squeeze(test_dataset.labels)

y_pred_probs = model.predict(x_test)

y_pred_classes = np.argmax(y_pred_probs, axis=1)

target_names = ['Basophil', 'Eosinophil', 'Erythroblast', 'Immature granulocyte', 'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

'''
print(classification_report(y_test, y_pred_classes, target_names=target_names))


cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Prediktált osztály', fontsize=12)
plt.ylabel('Valós osztály', fontsize=12)
plt.title('Tévesztési mátrix (Confusion Matrix)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''

unique_classes = np.unique(y_test)

model.layers[-1].activation = None

for cls in unique_classes:
    test_img_index = np.where(y_test == cls)[0][0]
    
    img_array = np.expand_dims(x_test[test_img_index], axis=0)
    true_label = y_test[test_img_index]

    heatmap = get_gradcam_plusplus(img_array, model, last_conv_layer_name='gradcam_target_layer')

    preds = model.predict(img_array, verbose=0)
    pred_label = np.argmax(preds[0])
    
    print("-" * 50)
    print(f"Osztály: {true_label} | Prediktált osztály: {pred_label}")
    
    display_gradcam(x_test[test_img_index], heatmap)

model.layers[-1].activation = keras.activations.softmax
