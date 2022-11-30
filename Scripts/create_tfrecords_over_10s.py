'''
Create TFRecordDataset for FBN project
Use in NeoCortex
Author: Xiaohui Zhang
Date Created: 03/07/2022
Data Updated: 09/06/2022
'''

# label wake=0, nrem=1, artifacts=2, rem=3
import os, glob, re, h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
import scipy
from scipy import stats
import scipy.io as sio

def _float_feature(value):
    '''Float 32
    '''
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int_feature(value):
    ''' Int 32
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def read_mat(fname):
    ''' Read matlab file
    '''
    f = h5py.File(fname, 'r')
    scores = np.array(f['scoringindex_file'][()], dtype=np.int32).T
    clean_scores_locs = np.squeeze(scores != 2)
    clean_scores = scores[clean_scores_locs]

    mask = np.array(f['xform_mask'][()], dtype=np.float32).T
    #mask = sio.loadmat('./midline_mask.mat')['papermask2']
    mask = np.reshape(mask, (128, 128, 1, 1)) 
    
    gcamp = np.array(f['data'][()], dtype=np.float32).T
    clean_gcamp = gcamp[:,:,:,clean_scores_locs]   
    clean_gcamp = np.multiply(gcamp, mask)
    return clean_gcamp, clean_scores, mask

def get_regexp(fname):
    m = re.match(r"(\d+)-(\w+\d+)-fc(\d+)-GSR_Ben.mat", os.path.basename(fname))
    name_parts = (m.groups())
    
    return int(name_parts[2])

def standardization(epoch_data, mask):
    """
    Standard Z-scoring on temporal dim/all dim
    """
    mask = np.reshape(mask, (128, 128, 1))
    mask = mask.astype('float')
    mask[mask==0] = np.nan
    masked_epoch = np.multiply(epoch_data, mask)
    zscored_epoch = stats.zscore(masked_epoch, axis=2, nan_policy='omit')
    zscored_epoch = np.nan_to_num(zscored_epoch)
    
    return zscored_epoch

def standardization_spatial(epoch_data, mask):
    """
    Standard Z-scoring on spatial dim
    """
    mask = np.reshape(mask, (128, 128, 1))
    mask = mask.astype('float')
    mask[mask==0] = np.nan
    masked_epoch = np.multiply(epoch_data, mask)
    masked_epoch = np.reshape(masked_epoch, (128*128, -1))
    zscored_epoch = stats.zscore(masked_epoch, axis=0, nan_policy='omit')
    zscored_epoch = np.nan_to_num(zscored_epoch.reshape((128, 128, 168)))
    
    return zscored_epoch


def min_max_normalization(epoch_data, mask):
    """
    Min-Max rescaling to [0,1]
    """
    mask = np.reshape(mask, (128, 128, 1))
    mask = mask.astype('float')
    mask[mask==0] = np.nan
    masked_epoch = np.multiply(epoch_data, mask)
    minmax_epoch = (masked_epoch - np.nanmin(masked_epoch)) / (np.nanmax(masked_epoch) - np.nanmin(masked_epoch))
    minmax_epoch = np.nan_to_num(minmax_epoch)
    
    return minmax_epoch

def create_tfrecord(data_path, des_path, mouse_list):

    for mouse_name in mouse_list:
        fnames = sorted(glob.glob(os.path.join(data_path, str(mouse_name), '*_Ben.mat')), key=lambda x: get_regexp(x))
        for fname in fnames:
            print(fname)           
            gcamp, scores, mask = read_mat(fname)
            num_epochs = len(scores)
            for epoch in range(num_epochs):
                with tf.io.TFRecordWriter(os.path.join(des_path, os.path.splitext(os.path.basename(fname))[0] + f"_epoch{epoch}.tfrecords")) as writer:
                    print(os.path.join(des_path, os.path.splitext(os.path.basename(fname))[0] + f"_epoch{epoch}.tfrecords"))
                    
                    new_frames_per_side = 42

                    if epoch == 0: 
                        epoch_data = np.concatenate((np.zeros((128,128,new_frames_per_side)), gcamp[:,:,:,epoch], gcamp[:,:,:new_frames_per_side,epoch+1]), axis=2)
                    elif epoch == num_epochs-1:
                        epoch_data = np.concatenate((gcamp[:,:,(168-new_frames_per_side):,epoch-1], gcamp[:,:,:,epoch], np.zeros((128,128,new_frames_per_side))), axis=2)
                    else:
                        epoch_data = np.concatenate((gcamp[:,:,(168-new_frames_per_side):,epoch-1], gcamp[:,:,:,epoch], gcamp[:,:,:new_frames_per_side,epoch+1]), axis=2)
                    
                    #print(epoch_data.shape)
                    
                    # z-score standardization
                    epoch_data = standardization(epoch_data, mask)
                    
                    epoch_label = scores[epoch]

                    if epoch_label == [3]:
                        print(epoch_label)
                        epoch_label = [2]
                    
                    feature = {'features': _float_feature(epoch_data.ravel()), 'label': _int_feature(epoch_label)}
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                
                writer.close()

def parse_tfrecord_fn(example):
   
    feature_description = {
            'features': tf.io.FixedLenFeature([128, 128, 252], tf.float32), 
            'label': tf.io.FixedLenFeature([], tf.int64)
            }
    example =  tf.io.parse_single_example(example, features=feature_description)
  
    return example['features'], example['label']


if __name__ == '__main__':
    data_path = '/shared/einstein/MRI/xiaohui/mouse_optical/sleep_stage/processed_data_sleep_3d/'
    des_path = '/shared/planck/MRI/xiaohui/mouse_optical/sleep_stage/2022-Ben-15s-raw-masked-tempzscore-broadband-states-tfrecords'
    mouse_list = [191030, 191114, 191115, 191204, 200115, 200127, 200128, 200204, 200313, 200402, 200813, 200814, 200910, 200925, 201002, 201125, 201215, 210108] 
    #mouse_list = [191030, 191114, 191115, 191204, 191211, 191218, 200115, 200127, 200128, 200204, 200313, 200402, 200813, 200814, 200910, 200925, 201002, 201125, 201215, 210108] 
    create_tfrecord(data_path, des_path, mouse_list) 

    #dataset = tf.data.TFRecordDataset('/shared/planck/MRI/xiaohui/mouse_optical/sleep_stage/2022-Ben-15s-raw-masked-tempzscore-broadband-states-tfrecords/191030-M5-fc1-GSR_Ben_epoch0.tfrecords')
    #parsed_dataset = dataset.map(partial(parse_tfrecord_fn))
    #
    #for x, y in parsed_dataset:
    #    x = x.numpy()
    #    y = y.numpy()
    #    print(y)
    #    plt.imshow(x[:, :, 200])
    #    plt.show()






