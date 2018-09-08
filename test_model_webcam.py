
import cv2
import numpy as np
import tensorflow as tf
from QueueVideoStream import WebcamVideoStream
import sys

    



data_format = tf.keras.backend.image_data_format()
def _preprocess_numpy_input(x, data_format, mode):
  """Preprocesses a Numpy array encoding a batch of images.

  Arguments:
      x: Input array, 3D or 4D.
      data_format: Data format of the image array.
      mode: One of "caffe", "tf" or "torch".
          - caffe: will convert the images from RGB to BGR,
              then will zero-center each color channel with
              respect to the ImageNet dataset,
              without scaling.
          - tf: will scale pixels between -1 and 1,
              sample-wise.
          - torch: will scale pixels between 0 and 1 and then
              will normalize each channel with respect to the
              ImageNet dataset.

  Returns:
      Preprocessed Numpy array.
  """
  if mode == 'tf':
    x /= 127.5
    x -= 1.
    return x

  if mode == 'torch':
    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  else:
    if data_format == 'channels_first':
      # 'RGB'->'BGR'
      if x.ndim == 3:
        x = x[::-1, ...]
      else:
        x = x[:, ::-1, ...]
    else:
      # 'RGB'->'BGR'
      x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None

  # Zero-center by mean pixel
  if data_format == 'channels_first':
    if x.ndim == 3:
      x[0, :, :] -= mean[0]
      x[1, :, :] -= mean[1]
      x[2, :, :] -= mean[2]
      if std is not None:
        x[0, :, :] /= std[0]
        x[1, :, :] /= std[1]
        x[2, :, :] /= std[2]
    else:
      x[:, 0, :, :] -= mean[0]
      x[:, 1, :, :] -= mean[1]
      x[:, 2, :, :] -= mean[2]
      if std is not None:
        x[:, 0, :, :] /= std[0]
        x[:, 1, :, :] /= std[1]
        x[:, 2, :, :] /= std[2]
  else:
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
      x[..., 0] /= std[0]
      x[..., 1] /= std[1]
      x[..., 2] /= std[2]
  return x

def train_preprocess(image):
    """ Image preprocessing for training

    Apply the following operations:
    - Horizontally flip the image with probability 1/2
    - Apply random brightness and saturation

    ToDo(Andrew): Apply same preprocessing as ImageNet, will be using pretrained weights
    """
    # if use_random_flip:
        # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image,max_delta=32.0/255)
    # image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
    
    # Make sure the image is still in [0,1]
    '''
    image = tf.image.random_brightness(image,max_delta=32.0/255)
    image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
    image= tf.clip_by_value(image,0.0,1.0)
    '''
    # image = tf.image.random_brightness(image,max_delta=32.0/255)
    # image = tf.image.random_saturation(image,lower=0.5,upper=1.5)

    image = _preprocess_numpy_input(image,data_format,'tf')
    return image

with tf.keras.utils.CustomObjectScope({'relu6': tf.nn.relu6 ,'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}):
    m = tf.keras.models.load_model("experiment/test/best_weights/after-epoch-1/model_acc_0.8611111111111112.h5")
    m.summary()
    # cam = cv2.VideoCapture(0)
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH,416)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT,416)

    # Q = Queue(maxsize=128)
    fvs = WebcamVideoStream().start()
    step=0
    while True:
        if step%10==0:
            img = fvs.read()
            k = cv2.resize(img,dsize=(224,224))
            k = k.astype(np.float32)
            k = train_preprocess(k)
            # cv2.imshow("Frame", k)

            k  = k[np.newaxis,...]
            res = m.predict(k,steps=1)
            # print(res)
            res=np.argmax(res,axis=1)[0]
            # display the size of the queue on the frame
            cv2.putText(img, str(res), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	        
        

            # show the frame and update the FPS counter
            cv2.imshow("Frame", img)
            if cv2.waitKey(1) == 27: 
                

                fvs.stop()  # esc to quit
                cv2.destroyAllWindows()
                break
        step+=1
    # while True:
        
    #     ret_val, img = cam.read()
    #     # img = cv2.resize(img,dsize=(416,416))

    #     # preprocess
    #     # print(img.shape)
    #     # k = img.copy()
    #     k = cv2.resize(img,dsize=(224,224))
    #     k = k.astype(np.float32)
    #     k = train_preprocess(k)
    #     # print(k.shape)
    #     # k = np.expand_dims(k,axis=0)
    #     k  = k[np.newaxis,...]
    #     # print(k.shape)
    #     res = m.predict(k,steps=1)
    #     # print(res)
    #     res=np.argmax(res,axis=1)[0]
    #     print(res)
    #     # boxes = yolo.predict(img)
    #     # image = draw_boxes(img, boxes, ["racoon","other"])
    #     # font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(img, str(res), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	        
    #     cv2.imshow('my webcam', img)
    #     if cv2.waitKey(1) == 27: 
    #         break  # esc to quit
