from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

CLASS_NAMES = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytophthora', 'Virus']

try:
    model = load_model('your_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(filepath):
    try: 
        img = image.load_img(filepath, target_size=(224, 224)) 
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return {
            'class': CLASS_NAMES[predicted_class],
            'confidence': confidence,
            'all_predictions': predictions[0].tolist()
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = None
            if model:
                prediction = predict_image(filepath)
                if prediction:
                    result = {
                        'class': prediction['class'],
                        'confidence': f"{prediction['confidence']*100:.2f}%",
                        'all_predictions': zip(CLASS_NAMES, prediction['all_predictions'])
                    }
            
            return render_template('index.html', 
                                 filename=filename,
                                 result=result if model and prediction else None)
    
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run('0.0.0.0' ,debug=True)