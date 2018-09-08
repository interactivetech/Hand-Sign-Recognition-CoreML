'''Split the SIGNS dataset into train/dev/test and resize images to 224x224.
build_dataset.py

1. Purpose of script is to resize the dataset
2. Split training into training and dev set
3. Describe statistics of training set, distribution of class labels
Original images are (3024, 3024)
We plan to reduce size by (224,224), as loading smaller images makes
training faster

Test set is already created, so splitting "train_signs" into train and dev sets
Want statistics on dev to be as representative as possible,
will take 20% of "train_signs" as dev set
'''

import argparse
import random
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def compute_statistics(args):
    '''
    Will take in a list of filenames, parse to get class labels
    and return distribution of labels
    '''
    train_data_dir = os.path.join(args.data_dir,'train_signs')
    test_data_dir = os.path.join(args.data_dir,'test_signs')
    print(train_data_dir)
    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir,f) for f in filenames if f.endswith('jpg')]
    tr_labels = np.array([int(f.split('/')[-1][0]) for f in filenames])

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir,f) for f in test_filenames if f.endswith('jpg')]
    test_labels = np.array([int(f.split('/')[-1][0]) for f in test_filenames])
    ax = sns.countplot(tr_labels)
    ax.set_title("Distribution of train labels")
    plt.show()
    ax=sns.countplot(test_labels)
    ax.set_title("Distribution of test labels")
    plt.show()
    return

SIZE=224

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default='data/SIGNS',
        help="Directory that contains the SIGNS dataset",
        )
parser.add_argument('--output_dir',default='data/224x224_SIGNS',help='Where to write the new data')
parser.add_argument('--compute_stats',default=0,help='Used to Compute stats')
def resize_and_save(filename,output_dir,size=SIZE):
    '''Resize the image contained in `filename` and save it to
    `output_dir`'''

    image = Image.open(filename)
    # Bilinear interpolation instead of nearest neighbor method
    image = image.resize((size,size),Image.BILINEAR)
    image.save(os.path.join(output_dir,filename.split('/')[-1 ]))
    return

if __name__=='__main__':

    # read and parse arguments
    args = parser.parse_args()
    if args.compute_stats==1:
        compute_statistics(args)
    else:
        # Check if data set argument passsed exists
        assert os.path.isdir(args.data_dir),"Couldnt find the dataset at {}".format(args.data_dir)

        # Define the data directories
        train_data_dir = os.path.join(args.data_dir,'train_signs')
        test_data_dir = os.path.join(args.data_dir,'test_signs')
        print(train_data_dir)
        # Get the filenames in each directory (train and test)
        filenames = os.listdir(train_data_dir)
        filenames = [os.path.join(train_data_dir,f) for f in filenames if f.endswith('jpg')]

        test_filenames = os.listdir(test_data_dir)
        test_filenames = [os.path.join(test_data_dir,f) for f in test_filenames if f.endswith('jpg')]
        # print(filenames)

        # Split the images into 'train_signs' into 80% train and 20% dev
        # Make sure to always shuffle with a fixed seed so that the split is reproducible

        random.seed(230)
        filenames.sort()
        random.shuffle(filenames)# strategy: shuffle and then split

        split = int(0.8*len(filenames))

        train_filenames = filenames[:split]
        dev_filenames = filenames[split:]

        filenames={
            'train':train_filenames,
            'dev':dev_filenames,
            'test':test_filenames
        }

        # make output dir if not exists
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        else:
            print("Warning: output_dir {} already exists".format(args.output_dir))

        # Preprocess train, dev, and test

        # Make all outputs for inside the output dir
        for split in ['train','dev','test']:
            output_dir_split=os.path.join(args.output_dir,'{}_signs'.format(split))
            if not os.path.exists(output_dir_split):
                os.mkdir(output_dir_split)
            else:
                print("Warning: dir {} already exists".format(output_dir_split))
            
            print("Processing {} data, saving preprocessed data to {}".format(split,
                output_dir_split))
            for filename in tqdm(filenames[split]):
                # print()
                resize_and_save(filename,output_dir_split,size=SIZE)
            
        print("Done building dataset")