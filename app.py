from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
#import pickle

app = Flask(__name__)
#cv = pickle.load(open("models/cv.pkl","rb"))
#clf = pickle.load(open("models/clf.pkl","rb"))
#model = load_model(r'models/modelnew.h5')
model = tf.keras.models.load_model(r'C:/Users/kbdsj/Desktop/Projects/OilyFaceDetection/dev/oily-face-care/models/modelnew.h5')
data_classes = ['Dry', 'Oil']

img_height = 224
img_width = 224

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files['image']
    #tokenized_email = cv.transform([email]) # X 
    #prediction = clf.predict(tokenized_email)
    #prediction = 1 if prediction == 1 else -1
    if uploaded_file is not None:
        image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.array_to_img(image_load)
        img_bat=tf.expand_dims(img_arr,0)

        predict = model.predict(img_bat)

        score = tf.nn.softmax(predict)
        predicted_class = data_classes[np.argmax(score)]
        #predicted_percentage = np.max(score) * 100 
    return render_template("index.html", prediction=predicted_class)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)