'''Evaluate Model'''

import argparse
import logging
import os

import tensorflow as tf
from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate_test

from model.utils import Params
from model.utils import set_logger

tf.enable_eager_execution()

# Add argument paraer
parser = argparse.ArgumentParser()
# Changing to directly load model saved after training
parser.add_argument('--model_dir',default='experiment/test',
                    help="Experiment directory containing params.json")
# parser.add_argument('--model_path',default='experiment/test/best_weights/after-epoch-15/model_acc_0.8695652173913043.h5'
parser.add_argument('--model_path',default='experiment/both_data_sources/best_weights/after-epoch-19/model_acc_0.9146341463414634.h5'

# experiment/test/best_weights/after-epoch-16
                    , help="Path to saved .h5 model")
parser.add_argument('--data_dir',default='data/224x224_SIGNS',
                    help='Subdirectory of model dir or file containing the weights')
parser.add_argument('--restore_from',default='best_weights',
                    help='Subdirectory of model dir or file containing the weights')    
if __name__=='__main__':


    # Set the random seed for the whole graph
    tf.set_random_seed(230)
    # Load the parameters
    args=parser.parse_args()
    json_path = os.path.join(args.model_dir,'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at{}".format(json_path)
    params = Params(json_path)
    # Set the logger

    set_logger(os.path.join(args.model_dir,'evaluate.log'))
    # Create the input data pipeline

    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir= os.path.join(data_dir,"test_signs")
    # Get the filenames from the test set
    test_filenames=os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir,f) for f in test_filenames if f.endswith('.jpg')]
    test_labels = [int(f.split('/')[-1][0]) for f in test_filenames]
    # specify the size of the evaluation set
    params.eval_size = len(test_filenames)
    # create the iterator over the dataset
    test_inputs = input_fn(False,test_filenames,test_labels,params)


    # Define the model: model_fn()
    logging.info("Creating the model...")
    with tf.keras.utils.CustomObjectScope({'relu6':tf.nn.relu6,'DepthwiseConv2D':tf.keras.layers.DepthwiseConv2D}):
        m_path = os.path.abspath(args.model_path) 
        print("Loading model from path {}".format(m_path))
        model= tf.keras.models.load_model(m_path)
        model.summary()
        accuracy = evaluate_test(model,test_inputs)
        print("Testing Accuracy {}".format(accuracy))
    # evaluate: evaluate()
