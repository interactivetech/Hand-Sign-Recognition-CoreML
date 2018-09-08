''' webcam_demo.py

Code includes loading keras model and running model prediction on webcam feed
where VideoStream.read() is grabbed in a seperate thread for reducing latency in 
Blocking I/O. 

Code adapted from pyimagesearch article post in WebcamVideoStream.py
'''

import tensorflow as tf 
import cv2
import argparse
import os
import numpy as np
from WebcamVideoStream import WebcamVideoStream
from model.utils import _preprocess_numpy_input


# parser arguments
parser = argparse.ArgumentParser()


# model_path
parser.add_argument("--model_path",default='experiment/test/best_weights/after-epoch-1/model_acc_0.8611111111111112.h5',
                    help="path to trained model")

# name_of_model

# author
# license

def load_model(model_path):

    with tf.keras.utils.CustomObjectScope({'relu6':tf.nn.relu6,'DepthWiseConv2D':tf.keras.layers.DepthwiseConv2D}):
        model = tf.keras.models.load_model(model_path)
        model.summary()
        return model

def preprocess_for_keras_model(image,data_format):
    '''
    Function handles resizing, preprocessing, and expanding dimension for keras model
    '''
    image = cv2.resize(image,dsize=(224,224))
    image = image.astype(np.float32)
    image = _preprocess_numpy_input(image,data_format,'tf')
    image = image[np.newaxis,...]# keras expect 4D tensor
    return image


if __name__=='__main__':
    
    args = parser.parse_args()

    data_format = tf.keras.backend.image_data_format()

    model = load_model(args.model_path)

    # instantiate WebcamVideoStream
    vs = WebcamVideoStream().start()
    step=0
    while True:
        if step%10==0:# used to reduce latency of demo, as model.predict() takes around 500ms

            img = vs.read()
            preprocessed_img = preprocess_for_keras_model(img,data_format)
            res = model.predict(preprocessed_img,steps=1)
            res = np.argmax(res)
            print(res)
            # add predicted class label to image to show result
            cv2.putText(img,str(res),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            cv2.imshow("Result",img)
            if cv2.waitKey(1)==27:
                vs.stop()
                break
    
        step+=1
    
    cv2.destroyAllWindows()


