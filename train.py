
''' train.py
Load hyperparameters, load iterators, and train model
'''
import argparse
import logging
import os
import random
import tensorflow as tf

# import keras
from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.evaluation import evaluate
from model.evaluation import get_missclassified_images
from model.model_fn import model_fn
from model.training import train_sess
from model.training import train_and_evaluate
import numpy as np

tf.enable_eager_execution()# Eager exec enabling

# Add Argument Parser
parser=argparse.ArgumentParser()
parser.add_argument('--model_dir',default='experiment/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir',default='data/224x224_SIGNS',
                    help='Directory containing the dataset')
parser.add_argument('--restore_from',default=None,
                    help='Directory of file containing weights to reload before training')







if __name__ == '__main__':

    # Set the random seed for the whole graph for reproducible experiments
    tf.set_random_seed(230)
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir,'params.json')
    # Check if path exists
    assert os.path.isfile(json_path),"No json configuration file is found at {}".format(json_path)
    params = Params(json_path)
    # Check that we are not overwriting some previous experiments
    # Comment these lines if you are developing a model and dont care about overwriting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir,"best_weights"))
    overwriting= model_dir_has_best_weights and args.restore_from is None
    # assert not overwriting, "Weights found in model_dir, aborting to avoid overwrite"
    # Set the logger
    set_logger(os.path.join(args.model_dir,'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir,"train_signs")
    dev_data_dir = os.path.join(data_dir,"dev_signs")
    # Get the filenames from the train and dev sets
    train_filenames = [os.path.join(train_data_dir,f) for f in os.listdir(train_data_dir)
                            if f.endswith('.jpg')]
    eval_filenames = [os.path.join(dev_data_dir,f) for f in os.listdir(dev_data_dir)
                        if f.endswith('.jpg')]
    # Labels will be between 0 and 5 (total of 6 classes)
    train_labels = [int(f.split('/')[-1][0]) for f in train_filenames ]
    eval_labels = [int(f.split('/')[-1][0]) for f in eval_filenames]


    # Sepecify the sizes of the dataset we train on and evaluate on
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)
    # Create the two iterators over the two decades: input_fn()
    train_inputs = input_fn(True,train_filenames,train_labels,params,NUM_EX=-1)
    eval_inputs = input_fn(False,eval_filenames,eval_labels,params,NUM_EX=-1)
    print(train_inputs)
    """ToDo(Andrew): Test the iterator fn call to validate correct output"""

    # Define the model: model_fn


    # Build model
    model = model_fn()
    model.summary()

    # # Create a Tensorflow optimizer, rather than using Keras version
    # # This is currently necessary when working in eager mode
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

    # # We will now compile and print out a summary of our model
    model.compile(loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])


    # Train the model: train_and_evaluate()

    BATCH_SIZE=128
    global_step = tf.train.get_or_create_global_step()
    summary_writer = tf.contrib.summary.create_file_writer(
        '224x224_logs/', flush_millis=10000)

    # Compute number of batches in one epoch (one full pass over the training set)
    num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
    # print(num_steps)
    '''python
    global_step = tf.train.get_or_create_global_step()
    summary_writer = tf.contrib.summary.create_file_writer(
        train_dir, flush_millis=10000)
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
      # model code goes here
      # and in it call
      tf.contrib.summary.scalar("loss", my_loss)
      # In this case every call to tf.contrib.summary.scalar will generate a record
      # ...
    '''


                
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
      # model code goes here
      # and in it call
      # In this case every call to tf.contrib.summary.scalar will generate a record
      # ...
        train_and_evaluate(model,train_inputs,eval_inputs,args)
        print("Done Training")
        get_missclassified_images(model,eval_inputs)
