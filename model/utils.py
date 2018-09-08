"""General utility functions"""
# ToDo(Andrew): Fininsh implementing Params class
import json
import logging

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


class Params():
    """Class that loads hyperparameters from json file"
    
    Example: 
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5# change the value of learning_rate in params
    ```
    """

    def __init__(self,json_path):
        self.update(json_path)

    def save(self,json_path):
        """Saves parameters to json file"""
        with open(json_path,'w') as f:
            json.dump(self.__dict__,f,indent=4)
    
    def update(self,json_path):
        """ Loads parameters from json file"""
        with open(json_path) as f:
            params=json.load(f)
            self.__dict__.update(params)


    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict`['learning_rate']"""
        return self.__dict__

def set_logger(log_path):
        """Sets the logger to log info in terminal and file `log_path`
        
        In general, it is useful to have a logger so that every output to the terminal
        is saved in a permanent file. Here we save it to model_dir/train.log

        Example:
        ```
        logging.info("Starting training...")
        ```

        Args:
        log_path: (string) where to log

        """

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

            # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))


def save_dict_to_json(d,json_path):
        """Saves dict of floats in json file
        
        Args:
            d: (dict) of floating-castable values(np.float,int,float,etc.)
            json_path: (string) path to json file
        """
        with open(json_path,'w') as f:
            # We need to convert the values to float for json
            # (it does not accept np.array, np.float)
            d={k:float(v) for k,v in d.items()}
            json.dump(d,f,indent=4)


    