"""Create the input data pipeline using  `tf.data`"""
import tensorflow as tf
from model.utils import _preprocess_numpy_input

data_format = tf.keras.backend.image_data_format()


def _parse_function(filename,label,size):
    """Obtain the image from the filename (for both training and validation).

    Following operations are applied:
    - Decode image from jpeg format
    - Convert to float and normalize to range [0,1]
    """
    #load image
    image_string = tf.read_file(filename)

    # Dont use the tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string,channels=3)
    # this will convert to float values in [0,1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    # convert labels
    # print(label)
    # label = float(label)
    #TODO(ANDREW): DO PREPROCESSING TO TRAIN FROM TRANSFER LEARNED IMAGENET
    resized_image = tf.image.resize_images(image,[size,size])


    return resized_image, label

def train_preprocess(image,label,use_random_flip):
    """ Image preprocessing for training

    Apply the following operations:
    - Horizontally flip the image with probability 1/2
    - Apply random brightness and saturation

    ToDo(Andrew): Apply same preprocessing as ImageNet, will be using pretrained weights
    """
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image,max_delta=32.0/255)
    image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
    
    # Make sure the image is still in [0,1]
    # image= tf.clip_by_value(image,0.0,1.0)

    # IMPORTANT TO DO CORRECT IMAGENET PREPROCESSING FOR MOBILENET WEIGHTS
    image = _preprocess_numpy_input(image,data_format,'tf')

    return image, label
def input_fn(is_training,filenames,labels,params,NUM_EX=-1):
    """Input function for SIGNS dataset.

    The filenames have format "{label}_IMG_{id}.jpg"
    ex: "data_dir/2_IMG_4548.jpg

    Args: is_training: (bool) whether to use the train or test pipeline
          Before training, we shuffle the data and have multiple epochs
    filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
    labels: (list) corresponding list of labels
    params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Check if filenames and labels have same length
    num_samples=len(filenames)
    if NUM_EX==-1:
        NUM_EX=num_samples
    print(len(labels),num_samples)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"


    # Create a dataset serving batches of images and labels
    # We dont repeat for multiple epochs because we always train and evaluate 
    # for one epoch
    parse_fn = lambda f,l: _parse_function(f,l,params.image_size)
    train_fn = lambda f,l:train_preprocess(f,l,params.use_random_flip)


    # if is_training, create tf.Data pipeline that uses multiple threads
    # prevent data starvation and better utilization of resources
    '''
    num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
            representing the number elements to process in parallel. If not
            specified, elements will be processed sequentially.
    '''
    if is_training:
        dataset=(tf.data.Dataset.from_tensor_slices((tf.constant(filenames[:NUM_EX]),tf.one_hot(labels[:NUM_EX],depth=6,dtype=tf.float32)  ))
        .shuffle(num_samples)# whole dataset inro the buffer ensures good shuffling
        .map(parse_fn,num_parallel_calls=params.num_parallel_calls)
        .map(train_fn,num_parallel_calls=params.num_parallel_calls)
        .batch(params.batch_size)
        .prefetch(1)# make sure you always have one batch ready to serve
        )

    else:
        dataset=(tf.data.Dataset.from_tensor_slices((tf.constant(filenames[:NUM_EX]),tf.one_hot(labels[:NUM_EX],depth=6,dtype=tf.float32)))
        .map(parse_fn)
        .batch(1)
        .prefetch(1)# make syre you always have one batch ready to serve
        )
    # Create reinitalizeable iterator from dataset
    """
    Creates an `Iterator` for enumerating the elements of this dataset.

    Note: The returned iterator will be initialized automatically.
    A "one-shot" iterator does not currently support re-initialization.

    Returns:
      An `Iterator` over the elements of this dataset.
    """
    # iterator = dataset.make_one_shot_iterator()
    # images,labels=iterator.get_next()
    # iterator_init_op = iterator.initializer

    # inputs = {'images':images,'labels':labels}

    return dataset

    