import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

model = load_model(r'models/modelnew.h5')
data_classes = ['Dry', 'Oil']

img_height = 224
img_width = 224

def app():
    st.header('Check Your Face Oiliness and Get Recommendations')
    # File uploader allows user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_load = tf.keras.utils.load_img(uploaded_file, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image_load)
        img_bat=tf.expand_dims(img_arr,0)

        predict = model.predict(img_bat)

        score = tf.nn.softmax(predict)
        st.image(uploaded_file, width=200)

        predicted_class = data_classes[np.argmax(score)]
        predicted_percentage = np.max(score) * 100  
        
        st.write('This face contains ' + predicted_class + 'ness')
        st.write('The {} percentage in the image is {:0.2f}%'.format(
            predicted_class, predicted_percentage))

        if predicted_class == 'Oil' and 100 >= predicted_percentage >= 95:
            st.write('Recommendations for oiliness level 4:')
            st.markdown('- Use oil-free products and blotting papers to control excess oil on the skin.')
            st.markdown('- Consider using a gentle exfoliant to unclog pores and prevent breakouts.')

        if predicted_class == 'Oil' and 95 > predicted_percentage >= 90:
            st.write('Recommendations for oiliness level 3:')
            st.markdown('- Use oil-free products and blotting papers to manage oil production.')
            st.markdown('- Incorporate products containing salicylic acid or benzoyl peroxide to help reduce acne.')

        if predicted_class == 'Oil' and 90 > predicted_percentage >= 85:
            st.write('Recommendations for oiliness level 2:')
            st.markdown('- Use lightweight, oil-free moisturizers to hydrate the skin without adding excess oil.')
            st.markdown('- Try using a clay mask once or twice a week to absorb excess oil and impurities.')

        if predicted_class == 'Oil' and 85 > predicted_percentage >= 80:
            st.write('Recommendations for oiliness level 1:')
            st.markdown('- Use a gentle foaming cleanser to remove excess oil and impurities without stripping the skin.')
            st.markdown('- Consider using a mattifying primer before applying makeup to help control shine throughout the day.')


        if predicted_class == 'Dry' and 100 > predicted_percentage >= 95:
            st.write('Recommendations for dryness level 4:')
            st.markdown('- Use a creamy, hydrating cleanser to gently cleanse the skin without causing further dryness.')
            st.markdown('- Apply a rich, emollient moisturizer immediately after bathing to lock in moisture.')

        if predicted_class == 'Dry' and 95 > predicted_percentage >= 90:
            st.write('Recommendations for dryness level 3:')
            st.markdown('- Incorporate a hyaluronic acid serum into your skincare routine to attract and retain moisture.')
            st.markdown('- Use a humidifier in your home to add moisture to the air, especially during dry weather.')

        if predicted_class == 'Dry' and 90 > predicted_percentage >= 85:
            st.write('Recommendations for dryness level 2:')
            st.markdown('- Limit the use of harsh, drying ingredients such as alcohol and fragrances in your skincare products.')
            st.markdown('- Apply a thick layer of moisturizing ointment or balm to extremely dry areas before bedtime.')

        if predicted_class == 'Dry' and 85 > predicted_percentage >= 80:
            st.write('Recommendations for dryness level 1:')
            st.markdown('- Drink plenty of water throughout the day to keep the skin hydrated from the inside out.')
            st.markdown('- Avoid taking long, hot showers which can strip the skin of its natural oils. Opt for lukewarm water instead.')

        if  80 > predicted_percentage :
            st.write('Your Face is healthy')

if __name__ == "__main__":
    app()
