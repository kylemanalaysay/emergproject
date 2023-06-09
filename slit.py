import streamlit as st
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('./model/TSR.h5')
  return model

st.title("Traffic Sign Detection System")

st.header("A Deep Learning Model to Detect Traffic Signs")

st.subheader("Built by Kyle Manalaysay, Chester Perez, and Harry Gayla")

st.markdown(
  "We're using the popular German Traffic Sign Recognition Benchmark dataset for Traffic Sign recognition. The model architecture combines convolutional layers for feature extraction and pooling layers for downsampling, followed by fully connected layers for classification. It also incorporates batch normalization and dropout for regularization and better training stability. We compile the model with an optimizer, loss function, and metrics. Then we train the model using the training data and evaluate its performance on the test data. Finally, we save the trained model using the save() method, specifying the desired file name ('TSR.h5' in this case)."
)

st.markdown(
  "The deployment of model in Web app, we select Streamlit Cloud Platform. We created our deep learning model and python script in Pycharm then commit and push it to GitHUB to be loaded in Streamlit app. Also installed the necessary packages and requirements for the deployment of Web app."
)           

import cv2
from PIL import Image,ImageOps
import numpy as np
def image_processing(image_path):
    model=load_model()
    data = []
    image = Image.open(image_path)
    image=ImageOps.fit(image,(30,30),Image.ANTIALIAS)
    data.append(np.array(image))
    X_test = np.array(data)
    predict_x = model.predict(X_test)
    Y_pred = np.argmax(predict_x, axis=1)
    return Y_pred

image_dir = "./For_Testing/"
image_files = os.listdir(image_dir)  
  
selected_image = st.selectbox("Select an image", image_files)
 
if selected_image:
    image_path = os.path.join(image_dir, selected_image)
    st.image(image_path,use_column_width=True)
    # Make prediction
    result = image_processing(image_path)
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
