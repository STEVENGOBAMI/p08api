import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, UpSampling2D, Activation
from tensorflow.python.keras.layers import Dropout, Reshape, Lambda
from keras.layers import BatchNormalization
# from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.python.keras.models import Model, load_model
from keras.utils import load_img, img_to_array
# from tensorflow.python.keras.utils import load_img, img_to_array
from tensorflow.python.keras.layers import Conv2DTranspose
from tqdm import tqdm
import cv2

import flask
from flask import Flask, request, render_template
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


model = load_model("model/um.h5", compile=False)
# model = unet()
# model.load_weights('model/w_unet.h5')

color_map = {
'0': [0, 0, 0],
 '1': [153, 153, 0],
 '2': [255, 204, 204],
 '3': [255, 0, 127],
 '4': [0, 255, 0],
 '5': [0, 204, 204],
 '6': [255, 0, 0],
 '7': [0, 0, 255]
}

app = Flask(__name__)

img_dir = "static/images"

@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'GET':

        return render_template('index.html')
        
    elif request.method=="POST":
        label = request.form.get('name')
        img_file = os.path.join(img_dir+"/"+label)
    
        img_real = load_img(img_file)
        img_real = img_to_array(img_real)
        alpha = 0.6
        dims = img_real.shape
        x = cv2.resize(img_real, (256, 256))
        x = np.float32(x)/255.
        z = model.predict(np.expand_dims(x, axis=0))
        z = np.squeeze(z)
        z = z.reshape(256, 256, 8)
        z = cv2.resize(z, (dims[1], dims[0]))
        y = np.argmax(z, axis=2)
    
        img_color = img_real.copy()   
        for i in range(dims[0]):
            for j in range(dims[1]):
                img_color[i, j] = color_map[str(y[i, j])]
        cv2.addWeighted(img_real, alpha, img_color, 1-alpha, 0, img_color)

    
        fig = plt.figure(figsize=(20, 10))
        plt.title(f'{label}')
        fig.add_subplot(1,3,1)
        plt.imshow(np.float32(img_real)/255.)
        fig.add_subplot(1,3,2)
        plt.imshow(y)
        fig.add_subplot(1,3,3)
        plt.imshow(np.float32(img_color)/255.)
        plt.savefig(f"static/predict_img/{label}", format='png')
        # plt.show()
        pic = f"static/predict_img/{label}"
        return render_template('index.html', image_mask=pic)

    return None

if __name__ == '__main__':
    app.run(debug=True)