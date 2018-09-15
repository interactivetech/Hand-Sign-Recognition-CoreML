'''Define model'''
import tensorflow as tf

# def build_model
def build_model(is_training,inputs,params):
    """ Compute Logits of the model
    (output distribution)
    ** this function is called in model_fn()

    Args:
      is_training: (bool) whether we are training or not
      inputs: (dict) contains the inputs of the graph (features, labels...)
              this can be `tf.placeholder` of outputs of `tf.data`
      params: (Params) hyperparameters

      Returns:
        x- output: (tf.Tensor) output of the model
        WILL RETURN KERAS MODEL WITH EAGER EXECUTION

    """
    # checking that size of input['images']==(None,params.image_size,params.image_size,3)

    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool

    # can load mobilenet, with pretrained weights


    
    return

# def model_fn(mode,inputs,params,reuse=False):
#     """Model function defining graph operations

#     Args:
#       mode: (string) can be 'train' or 'eval'
#       inputs: (dict) contains the inputs of the graph (features, labels...)
#               this can be a `tf.placeholder` of outputs `tf.data`
#       params:(Params) contains hyperparameters of the model (ex:`params.learning_rate`)
#       reuse:(bool) whether to reuse the weights

#     Returns:
#       model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
#     """

#     ## MODEL: define the layers of the model

#     # Define loss and accuracy 

#     # Define training step that minimizes the loss with the Adam Optimizer
    
#     ## METRICS AND SUMMARIES
#     # Metrics for evaluation using tf.metrics (average over whole dataset)
    
#     ## MODEL SPECIFICATION
#     '''
#     Create a model specification and return it
#     It contains nodes or operations in the graph that will be used for training and evaluation
#     '''
#     return

def model_fn(NUM_CAT=6):
    # input = tf.keras.layers.Input(shape=(224,224,3))
    m = tf.keras.applications.MobileNet(include_top=False,weights='imagenet',input_shape=(224,224,3),
          )
    # m.summary() 

    # x = m(input)# (None, 7, 7, 1024)
    x = m.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Dense(512,activation=tf.nn.relu)(x)
    # x = tf.keras.layers.Dense(NUM_CAT,activation=tf.nn.softmax)(x)
    x = tf.keras.layers.Reshape((1, 1, int(1024)), name='reshape_1')(x)
    x = tf.keras.layers.Dropout(1e-3, name='dropout')(x)
    x = tf.keras.layers.Conv2D(NUM_CAT, (1, 1),
                          padding='same',
                          name='conv_preds')(x)
    x = tf.keras.layers.Activation('softmax', name='act_softmax')(x)
    x = tf.keras.layers.Reshape((NUM_CAT,), name='reshape_2')(x)
    model = tf.keras.models.Model(inputs=m.input,outputs=x)
    return model
