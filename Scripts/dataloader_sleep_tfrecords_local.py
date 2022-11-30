"""
Dataloader for sleep stage classification on GCAMP data
using TFRecords using on local machine
Author: Xiaohui Zhang
Date created: 03/18/2022
"""

import os, glob, json, re
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def add_arguments(parser):
    parser.add_argument('--data_path', type=str, default='/shared/einstein/MRI/xiaohui/mouse_optical/sleep_stage/2022-Ben-10s-masked-broadband-states-tfrecords')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size in training')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classess in classification branch')
    parser.add_argument('--num_frames', type=int, default=168, help='number of input image frames')
    parser.add_argument('--hemispheric', action='store_true', help='whether to use hemispheric data')
    parser.add_argument('--loss', type=str, default='cross_entropy', help='Training loss')

    return parser

class dataloader_sleep_tfrecords:
    """
    3D Sleep Classification Dataset Class
    """
    def __init__(self, project=None):
        self.project = project
        if self.project.params.hemispheric:
             self.project.params.image_shape = [self.project.params.num_frames, 128, 64, 1]
        else:
             self.project.params.image_shape = [self.project.params.num_frames, 128, 128, 1]

    def load_data(self):
        # speficy the data split by keeping an independent mouse
        with open("mouse_split_ben.json", 'r') as file:
            mouse_list = json.load(file)
               
        self.project.params.mouse_list = mouse_list

        tf_fnames = []
        for train_mice in mouse_list['train']:
            tf_fnames.extend(glob.glob(os.path.join(self.project.params.data_path, f"{str(train_mice)}*.tfrecords")))
        tf_fnames = sorted(tf_fnames)
        
        #tf_fnames = sorted(glob.glob(os.path.join(self.project.params.data_path, f"*.tfrecords")))
        train_fnames, tmp_fnames = train_test_split(tf_fnames, test_size=0.2, random_state=self.project.params.seed, shuffle=True)
        val_fnames, test_fnames = train_test_split(tmp_fnames, test_size=0.5, random_state=self.project.params.seed, shuffle=True)
        
        if self.project.params.mode == "test_subjectwise" or self.project.params.mode == "gradcam" or self.project.params.mode == "attention_weights":
            test_fnames = sorted(glob.glob(os.path.join(self.project.params.data_path, f"{str(mouse_list['test'])}*.tfrecords")), key=lambda x: get_regexp(x))
        
        print(len(train_fnames), len(val_fnames), len(test_fnames))
        
        train_dataset = self.load_dataset(train_fnames, reshuffle_each_iteration=True)
        val_dataset = self.load_dataset(val_fnames, reshuffle_each_iteration=False)
        test_dataset = self.load_dataset(test_fnames, reshuffle_each_iteration=False)
        
        #for x, y in val_dataset:
        #   print(y.numpy()[0])
        #   print(x.numpy().shape)
        #   plt.imshow(x.numpy()[0,100,:,:,0])
        #   plt.show()
 
        return  train_dataset, val_dataset, test_dataset

    def load_dataset(self, fnames, reshuffle_each_iteration=None):
        
        def _parse_records(record):
            feature_description = {
                'features': tf.io.FixedLenFeature([128, 128, self.project.params.num_frames, 1], tf.float32), 
                'label': tf.io.FixedLenFeature([], tf.int64)
                }
            example = tf.io.parse_single_example(record, features=feature_description)
            return example['features'], example['label']
         
        def _process_dataset(features, label):
            features = tf.transpose(features, perm=[2,0,1,3])
            if self.project.params.num_frames < 168:
                print('Using less than 10s data...\n')
                features = features[:self.project.params.num_frames, ]
            if self.project.params.hemispheric:
                print('Using just hemispheric data...\n')
                features = features[:, :, :64, :]
            if self.project.params.num_classes == 2 and label == tf.constant(2, dtype=tf.int64):
                label = tf.constant(1, dtype=tf.int64)
            
            #if "poly" in self.project.params.loss:
            #    label = tf.one_hot(label, depth=3)
            return features, label
       
        dataset = tf.data.TFRecordDataset(fnames) #num_parallel_reads=16)
        dataset = dataset.map(map_func=_parse_records, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(map_func=_process_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if self.project.params.mode == "train":
            dataset = dataset.shuffle(buffer_size=32, seed=self.project.params.seed, reshuffle_each_iteration=reshuffle_each_iteration)
        dataset = dataset.batch(batch_size=self.project.params.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        return dataset

def get_regexp(fname):
    m = re.match(r"(\d+)-(\w+\d+)-fc(\d+)-GSR_Ben_epoch(\d+).tfrecords", os.path.basename(fname))
    name_parts = (m.groups())
    return int(name_parts[2]), int(name_parts[3])


