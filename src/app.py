import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model



model_path = "model\covid_detection_model.h5"  
model = tf.keras.models.load_model(model_path)


def predict_image(img):
    img_resized = tf.image.resize_with_pad(img, target_height=224, target_width=224)
    img_array = image.img_to_array(img_resized)
    
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array_copy = np.copy(img_array)
    img_array_copy = preprocess_input(img_array_copy)

    prediction = model.predict(img_array_copy)
    return prediction[0][0]


st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon=":microbe:",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("COVID-19 Detection from CT Scan Images")
st.markdown(
    """
    Upload your CT scan image, and we will predict whether it indicates COVID-19.
    *This is not a substitute for professional medical advice.*
    """
)


uploaded_file = st.file_uploader("Choose a CT scan image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    st.subheader("Uploaded Image:")
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)


    if st.button("Predict"):
        image_to_predict = Image.open(uploaded_file)
        prediction = predict_image(image_to_predict)

       
        st.subheader("Prediction Result:")
        result_card = st.empty()
        with st.spinner("Analyzing..."):
            st.write("") 
            if prediction > 0.5:
                result_card.error(f"COVID-19 (Probability: {prediction:.2f}): COVID POSITIVE\n Please contact to the doctor ASAP:\n Call: 108")
                
            else:
                result_card.success(f"Non-COVID-19 (Probability: {1 - prediction:.2f}): COVID NEGATIVE")

        
        st.write("") 
        st.subheader("Additional Information:")
        st.info(
            """
            - The model predictions are based on the analysis of CT scan images.
            - This is not a substitute for professional medical advice.
            """
        )

# Footer
st.sidebar.markdown(
    """
    * [GitHub Repo](https://github.com/yourusername/your-repo)
    * [Report Issues](https://github.com/yourusername/your-repo/issues)
    """
)
