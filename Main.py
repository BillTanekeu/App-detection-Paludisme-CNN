#import tensorflow as tf
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras.models import load_model
#from tensorflow.image import per_image_standardization

st.set_page_config(page_title = "Dep'sMalaria", page_icon = 'Images/logo.png', layout="wide")
st.image('Images/bant.png')
st.header("Malaria case detection using blood cell pictures.")
targets_names = ["Uninfected","Parasitized"]
model = load_model("cnn_malaria")
im_size = 28
im_prof = 3





with open("style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html= True)
#st.markdown(" <style> body{background-image: url(image7.jpg);} </style>", unsafe_allow_html= True )
st.sidebar.image("Images/malaria.png")
st.sidebar.header("Thank you for using Dep'sMalaria!")
st.sidebar.markdown("<h2> Contact us:</h2>", unsafe_allow_html=True)
c1, c2, c3 = st.sidebar.columns(3)
c1.markdown("<a href =mailto:caterbilljordan.com><img src='https://img.icons8.com/officel/31/000000/gmail-login.png'/> </a> ", unsafe_allow_html=True)
c2.markdown("<a href =https://twitter.com/JTanekeu><img src='https://img.icons8.com/color/31/000000/twitter--v1.png'/></a> ", unsafe_allow_html=True)
c3.markdown("<a href =https://t.me/Tanekeu> <img src='https://img.icons8.com/color/31/000000/telegram-app--v1.png'/> </a> ", unsafe_allow_html=True)



st.snow()
def Main():
    file = st.file_uploader("Choose a picture", type = ["jpeg", "png", "jpg"])
    if file is not None:
        image = Image.open(file)
        figure = plt.figure(figsize = (5,5))
        plt.imshow(image)
        plt.axis("off")
        result = prediction(image)
        col1, col2= st.columns(2)
        col1.metric("Predicted class:", result["target_name"] )
        col2.metric("Probability:",  "{}%".format(truncate(result["probability"]*100,2)))
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
