import streamlit as st 
import tensorflow as tf 
from PIL import Image
import numpy as np
import cv2

#load model
model=tf.keras.models.load_model("ann_model.keras",compile=False)
print(model.input_shape) 
#title 
st.title ("Digit Recognition System Using ANN")

#upload file 

uploaded_file=st.file_uploader("choose image",type=["jpg","jpeg","png"])

#check if image is uploaded or not 
if uploaded_file is not None:
    #open image 
    image=Image.open(uploaded_file).convert('L') #L stand for luminance
    #image will contain only shades of gray 
    #display image 
    st.image(image,caption="uploaded image ",width=150)

    #convert image into array 
    img=np.array(image)
    #resize to 28x28
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)

    # Invert colors
    img = 255 - img

    # Normalize
    img = img.astype("float32") / 255.0
    #reshape for prediction 
    img=img.reshape (1,28,28)
    st.image((img.reshape(28,28) * 255).astype(np.uint8), width=150)
    #prediction
    prediction=model.predict(img)
    predicted_digit =np.argmax(prediction)
    st.success(f"Expected digit : {predicted_digit}")