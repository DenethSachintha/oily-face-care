import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np


model = load_model(r'models/modelnew.h5')
data_classes = ['Dry', 'Oil']

img_height = 224
img_width = 224




def app():
    st.header('Check Your Face Oilyness and Get Recommendations')
    # File uploader allows user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.array_to_img(image_load)
        img_bat=tf.expand_dims(img_arr,0)

        predict = model.predict(img_bat)

        score = tf.nn.softmax(predict)
        st.image(uploaded_file, width=200)

        predicted_class = data_classes[np.argmax(score)]
        predicted_percentage = np.max(score) * 100  
    
        st.write('This face contains ' + predicted_class + 'ness')
        st.write('The {} percentage in the image is {:0.2f}%'.format(
            predicted_class, predicted_percentage))


if __name__ == "__main__":
    app()
        



