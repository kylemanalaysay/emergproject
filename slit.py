import streamlit as st
import tensorflow as tf
from werkzeug.utils import secure_filename

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('./model/TSR.h5')
  return model
st.write("""
# Traffic Sign Detection System"""
)
file=st.file_uploader("Choose plant photo from computer",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def image_processing(img):
    model=load_model()
    data = []
    image=ImageOps.fit(img,(30,30),Image.ANTIALIAS)
    data.append(np.array(image))
    X_test = np.array(data)
    predict_x = model.predict(X_test)
    Y_pred = np.argmax(predict_x, axis=1)
    return Y_pred

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    # Make prediction
    result = image_processing(image)
    classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Vehicle > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing vehicle > 3.5 tons' }
    s = [str(i) for i in result]
    a = int("".join(s))
    result="Predicted Traffic🚦Sign is: "+ classes[a]
    
    st.success(result)
