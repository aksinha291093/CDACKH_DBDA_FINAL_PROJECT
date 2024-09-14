from flask import Flask, render_template, request, url_for
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd 

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_name = db.Column(db.String(100))
    mobile_number = db.Column(db.String(20))
    item_name = db.Column(db.String(50))
    quantity = db.Column(db.Float)
    price_per_kg = db.Column(db.Float)
    total_price = db.Column(db.Float)
    date = db.Column(db.DateTime, default=datetime.now) 

with app.app_context():
    db.create_all()

labels_prices = {
    'apple': 140.0, 'banana': 50.0, 'beetroot': 100.0, 'bell pepper': 150.0, 'cabbage': 90.0,
    'capsicum': 130.0, 'carrot': 100.0, 'cauliflower': 140.0, 'chilli pepper': 230.0, 'corn': 120.0,
    'cucumber': 90.0, 'eggplant': 120.0, 'garlic': 200.0, 'ginger': 200.0, 'grapes': 160.0,
    'jalepeno': 250.0, 'kiwi': 150.0, 'lemon': 70.0, 'lettuce': 70.0, 'mango': 180.0,
    'onion': 80.0, 'orange': 80.0, 'paprika': 210.0, 'pear': 100.0, 'peas': 80.0,
    'pineapple': 200.0, 'pomegranate': 220.0, 'potato': 50.0, 'raddish': 110.0, 'soy beans': 160.0,
    'spinach': 120.0, 'sweetcorn': 150.0, 'sweetpotato': 110.0, 'tomato': 100.0, 'turnip': 130.0,
    'watermelon': 120.0
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/result", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        name = request.form['name']
        mobile = request.form['mobile']
        image = request.files['image']
        quantityform = float(request.form['quantity'])

        if image:
            # Save the image to the upload folder
            image_filename = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_filename)

            # Create the URL for the uploaded image
            image_url = url_for('static', filename=f'uploads/{image.filename}')

        predict = answer(image_filename)
        price_kg = labels_prices[predict]
        total_price=price_kg*quantityform
        
        new_prediction = Prediction(
            customer_name=name,
            mobile_number=mobile,
            item_name=predict,
            quantity=quantityform,
            price_per_kg= price_kg,
            total_price=price_kg*quantityform
        )

        db.session.add(new_prediction)
        db.session.commit()

        return render_template('result.html', name=name, mobile=mobile, image=image_url,predict=predict,quantity=quantityform,price_per_kg=price_kg,total_price=total_price)
    
def answer(image_url):
    test_set = tf.keras.utils.image_dataset_from_directory(
    'D:/veg/test',
    labels = 'inferred',
    label_mode= 'categorical',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(64,64),
    shuffle= True,
    seed=None,
    validation_split=None,
    subset= None,
    interpolation = "bilinear",
    follow_links= False,
    crop_to_aspect_ratio = False
    )
    
    cnn = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(image_url,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Comverting single image into batch  because our model build in 2d array
    prediction = cnn.predict(input_arr)
    result_index = np.where(prediction[0] == max(prediction[0]))
    
    return test_set.class_names[result_index[0][0]]


if __name__=="__main__":
    app.run(debug=True)