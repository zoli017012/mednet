from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import tensorflow as tf
import numpy as np
from xai import get_gradcam_plusplus, get_gradcam
import matplotlib.cm as cm

app = Flask(__name__)
CORS(app) # Engedélyezi, hogy a weboldal kommunikáljon a szerverrel
model = tf.keras.models.load_model('mednet_modell_128_31.keras')
class_names = ['Basophil', 'Eosinophil', 'Erythroblast', 'Immature granulocyte', 'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

def analyze_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)


    heatmap = get_gradcam_plusplus(img_array, model, last_conv_layer_name='gradcam_target_layer')

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(img.size, Image.Resampling.BILINEAR)
    heatmap_resized_array = np.array(heatmap_img) / 255.0
    
    # Színtérkép ('jet') alkalmazása a kiszámolt értékeken
    cmap = cm.get_cmap('jet')
    heatmap_colored = cmap(heatmap_resized_array) # Ez (R, G, B, Alpha) csatornákat ad
    
    # Csak az RGB színeket tartjuk meg, és 0-255 skálára konvertáljuk
    heatmap_colored = np.uint8(255 * heatmap_colored[:, :, :3])
    heatmap_pil = Image.fromarray(heatmap_colored)
    
    # Rávetítés az eredeti képre. Az alpha=0.4 jelenti a 40%-os átlátszóságot.
    result_image = Image.blend(img, heatmap_pil, alpha=0.4)

    return f"{class_names[np.argmax(predictions)]}", result_image


@app.route('/process-image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Nincs fájl feltöltve'}), 400
        
    image_bytes = request.files['file'].read()
    
    # Itt hívjuk a függvényt, és azonnal szét is bontjuk a két változóba!
    prediction_text, generated_img = analyze_image(image_bytes)
    
    # A kapott képet (PIL formátum) Base64 szöveggé alakítjuk a webes küldéshez
    buffered = io.BytesIO()
    generated_img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Csomagolás JSON-be és küldés a frontendnek
    return jsonify({
        'prediction': prediction_text,
        'image_data': img_base64
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)