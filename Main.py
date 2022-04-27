#import tensorflow as tf
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.image import per_image_standardization

st.header("Classification d'images")
targets_names = ["Uninfected","Parasitized"]
model = load_model("cnn_malaria")
im_size = 28
im_prof = 3

def Main():
    file = st.file_uploader("Choisir une image", type = ["jpeg", "png", "jpg"])
    if file is not None:
        image = Image.open(file)
        figure = plt.figure(figsize = (5,5))
        plt.imshow(image)
        plt.axis("off")
        result = prediction(image)
        st.write("classe predicte:", result["target_name"] )
        st.write("probabilité:", truncate(result["probability"]*100,2),"%")
        st.pyplot(figure)

def prediction(image):
    image = pretraitement(image)
    prev =  model.predict(image)
    prob = prev[0][np.argmax(prev)]
    class_name = targets_names[np.argmax(prev)]
    
    result = {"target_name": class_name, "probability": prob}
    return result

def pretraitement(image):
    image = image.resize((im_size, im_size))
    image = np.array(image)
    image = image/255
    image = image.reshape(1,im_size,im_size,im_prof)
    return image

def truncate(num, n):
    integer = int(num*(10**n))/(10**n)
    return float(integer)

Main()
