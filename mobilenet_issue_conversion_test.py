# reproducing issue
'''
Code that fails
'''
# import tensorflow as tf 
# import coremltools

# model = tf.keras.applications.MobileNet(weights=None)
# model.load_weights("~/.keras/models/mobilenet_1_0_224_tf.h5")
# model.summary()

# fcn_mlmodel = coremltools.converters.keras.convert(
#         model,
#         input_names = 'image',
#         image_input_names = 'image',
#         output_names = 'class_label',
#         image_scale = 1/127.5,
#         red_bias=-1.0,
#         blue_bias=-1.0,
#         green_bias=-1.0
#     )
# fcn_mlmodel.author ="Bill Bob"
# fcn_mlmodel.license="MIT"
# fcn_mlmodel.short_description="Outputs Hand Sign class given input image"
# fcn_mlmodel.input_description['image']="Image size (224,224,3)"
# fcn_mlmodel.output_description['class_label']=" Class label"
# fcn_mlmodel.save("Test_Mobilenet.mlmodel")

'''
Code that works
'''
import keras
import coremltools
model = keras.applications.MobileNet(weights=None)
model.load_weights("/Users/andrewmendez1/.keras/models/mobilenet_1_0_224_tf.h5")
model.summary()

fcn_mlmodel = coremltools.converters.keras.convert(
        model,
        input_names = 'image',
        image_input_names = 'image',
        output_names = 'class_label',
        image_scale = 1/127.5,
        red_bias=-1.0,
        blue_bias=-1.0,
        green_bias=-1.0
    )
fcn_mlmodel.author ="Bill Bob"
fcn_mlmodel.license="MIT"
fcn_mlmodel.short_description="Outputs Hand Sign class given input image"
fcn_mlmodel.input_description['image']="Image size (224,224,3)"
fcn_mlmodel.output_description['class_label']=" Class label"
fcn_mlmodel.save("Test_Mobilenet.mlmodel")