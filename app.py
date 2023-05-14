import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename
from  PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding', False)

model = pickle.load(open("image_classification_model.pkl","rb"))

html_temp = """
    <div class="" style="background-color:blue;">
    <div class="clearfix">
    <div class="col-md-12">
    <center><p style="font-size:40px;color:white;margin-top:10px;">Workshop on </p></center>
    <center><p style="font-size:40px;color:white;margin-top:10px;">Artificial Intelligence & Data Science </p></center>
    </div>
    </div>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
st.title("""
        Digit Recognition
         """
         )
file= st.file_uploader("Please upload image", type=("jpg", "png"))

# def import_and_predict(image_data):
#   single_test = image_data[:, :, 0]
#   single_test = single_test.reshape(1,-1)
#   prediction = int(model.predict(single_test))
#   return prediction

def import_and_predict(image_data):
    # Resize the image to 28x28 pixels and convert to grayscale
    size = (8, 8)
    image = ImageOps.grayscale(image_data.resize(size))
    # Convert the image to a 1D array and normalize the pixel values
    pixel_values = np.array(image).reshape(1, -1) / 255.0
    # Make the prediction using the loaded model
    prediction = int(model.predict(pixel_values))
    return prediction

if file is None:
  st.text("Please upload an Image file")
else:
  image=Image.open(file)
  image_np=np.array(image)
  st.image(image_np,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Predict Digit"):
  result=import_and_predict(image_np)
  st.success('Model has predicted the image is of  {}'.format(result))
if st.button("About"):
  st.header("Rohan Kandpal")
  st.subheader("Student, Poornima Group of Institution")
  
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:20px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
