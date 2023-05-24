import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('./model/TSR.h5')
  return model
model=load_model()
st.write("""
# Traffic Sign Detection System"""
)
file=st.file_uploader("Choose plant photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(30,30)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Speed limit (20km/h)','Speed limit (30km/h)',
                 'Speed limit (50km/h)',
                 'Speed limit (60km/h)','Speed limit (70km/h)',
                 'Speed limit (80km/h)','End of speed limit (80km/h)',
                 'Speed limit (100km/h)','No passing',
                 'No passing veh over 3.5 tons','Right-of-way at intersection',
                 'Priority road','Yield',
                 'Stop','No vehicles',
                 'Vehicle > 3.5 tons prohibited',
                 'No entry', 'General caution',
                 'Dangerous curve left','Dangerous curve right',
                 'Double curve','Bumpy road',
                 'Slippery road','Road narrows on the right',
                 'Road work','Traffic signals',
                 'Pedestrians','Children crossing',
                 'Bicycles crossing',
                 'Beware of ice/snow','Wild animals crossing','End speed + passing limits',
                 'Turn right ahead','Turn left ahead','Ahead only',
                 'Go straight or right','Go straight or left',
                 'Keep right','Keep left',
                 'Roundabout mandatory','End of no passing',
                 'End no passing vehicle > 3.5 tons']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
