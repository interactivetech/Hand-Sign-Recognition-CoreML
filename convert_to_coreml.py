'''convert_to_coreml.py

This function converts a keras model that was trained using this repo to CoreML
'''
import tensorflow as tf 
import coremltools
import argparse
import keras
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model.utils import _preprocess_numpy_input

# print(dir(tf.keras.applications.mobilenet.MobileNet))
from tensorflow.python.keras.applications import mobilenet
# print(dir(mobilenet))
# from keras.applications.mobilenet import DepthwiseConv2D# Tensorflow 1.5 does not have built in
# 2.01b version

parser=argparse.ArgumentParser()
# model_path
parser.add_argument("--model_path",default="experiment/test/best_weights/after-epoch-1/model_acc_0.8611111111111112.h5",
                    help="path to train .h5 keras model")
# author
parser.add_argument("--author",default="Andrew Mendez",
                    help="Name of Author that will be saved in CoreML model")
# license
parser.add_argument("--license",default='Copyright @Andrew Mendez 2018')
# name_of_coreml_model
parser.add_argument("--name_of_coreml_model",default='Hand_Sign_Recognition.mlmodel',
                    help="Name of CoreML model")

def model_fn(NUM_CAT=6):
    input = keras.layers.Input(shape=(224,224,3))
    m = keras.applications.MobileNet(include_top=False,weights=None)
    # m.summary() 

    x = m(input)# (None, 7, 7, 1024)
    x = keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Dense(512,activation=tf.nn.relu)(x)
    # x = tf.keras.layers.Dense(NUM_CAT,activation=tf.nn.softmax)(x)
    x = keras.layers.Reshape((1, 1, int(1024)), name='reshape_1')(x)
    x = keras.layers.Dropout(1e-3, name='dropout')(x)
    x = keras.layers.Conv2D(NUM_CAT, (1, 1),
                          padding='same',
                          name='conv_preds')(x)
    x = keras.layers.Activation('softmax', name='act_softmax')(x)
    x = keras.layers.Reshape((NUM_CAT,), name='reshape_2')(x)
    model = keras.models.Model(inputs=input,outputs=x)
    return model
def load_model(model_path):
    # print(dir(tf.keras.applications.mobilenet.MobileNet))
    # tf.keras.models.mobilenet.re
    # with keras.utils.generic_utils.CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    # with tf.keras.utils.CustomObjectScope(custom_objects={ 'relu6': tf.keras.applications.mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D}):
    
    # 
    model2 = tf.keras.models.load_model(model_path, custom_objects={ 'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    model=model_fn()# Keras Model!
    #
    model.load_weights(model_path)# Transfering weights from tf.keras to keras

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
    
    keras_model = load_model(args.model_path)
    keras_model.summary()
    fcn_mlmodel = coremltools.converters.keras.convert(
        keras_model,
        input_names = 'image',
        image_input_names = 'image',
        output_names = 'class_label',
        class_labels=["0","1","2","3","4","5"],
        image_scale = 1/127.5,
        red_bias=-1.0,
        blue_bias=-1.0,
        green_bias=-1.0
    )
    fcn_mlmodel.author =args.author
    fcn_mlmodel.license=args.license
    fcn_mlmodel.short_description="Outputs Hand Sign class given input image"
    fcn_mlmodel.input_description['image']="Image size (224,224,3)"
    fcn_mlmodel.output_description['class_label']=" Class label 0-5"
    fcn_mlmodel.save(args.name_of_coreml_model)

    model = coremltools.models.MLModel("Hand_Sign_Recognition_224_0.85.mlmodel")
    img =Image.open("/Users/andrewmendez1/Documents/Hand Sign Recognition/data/224x224_SIGNS/dev_signs/0_IMG_5864.jpg")

    # data_format = tf.keras.backend.image_data_format()
    # img = preprocess_for_keras_model(img,data_format)
    
    '''
    https://apple.github.io/coremltools/generated/coremltools.models.MLModel.html

    'has_key', 'items', 'iteritems', 'iterkeys', 'itervalues', 'keys', 'pop',
     'popitem', 'setdefault', 'update', 'values', 'viewitems', 'viewkeys', 'viewvalues'
    
    Properties:
    {
        classLabel:
        class_label:{...confidence scores}
    }
    '''
        #ToDo(Andrew): Make sure this model correctly predicts image from training set
    res = model.predict({"image":img})

    img = np.asarray(img)
    plt.imshow(img)
    plt.show()
    print(res["classLabel"], res["class_label"][res["classLabel"]])

