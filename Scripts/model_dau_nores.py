"""
CNN-LSTM model function for local machine
Created date: 03/18/2022
Author: Xiaohui Zhang
"""

import os, json
import argparse
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from focal_loss import SparseCategoricalFocalLoss, BinaryFocalLoss

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from AttentionLayer import CustomAttentionTF, LSTMAttentionLayer, SimAM, ChannelAttention, SpatialAttention
from PolyLoss import poly1_cross_entropy, poly1_focal_cross_entropy
from gradcam import gradcam, gradcam_plus
from visualize_lstm_attention_weights import visualize_lstm_attention_weights
from utils import calculate_metrics
#mixed_precision.set_global_policy('mixed_float16')

def add_arguments(parser):
    parser.add_argument('--num_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--lr_init', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--model_savedir', type=str, default='../Models', help='path to save trained models')
    parser.add_argument('--logs_dir', type=str, default='../Logs', help='path to save trained models')
    parser.add_argument('--model_continue', action='store_true', default=False, help='reload model and continue training')
    parser.add_argument('--result_savedir', type=str, default='../Results/encoded_components/', help='path to save predicted results')
    parser.add_argument('--result_action', type=str, default=None, help="Plot the result or save it as variables/images")
    parser.add_argument('--num_rnn_units', type=int, default=None, help="Number of RNN units in one-directional LSTM")
    parser.add_argument('--gradcam_label', type=int, default=0, help="GradCAM maps of sleep state")
    return parser

class model:
    def __init__(self, project):
        self.project = project
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath = os.path.join(self.project.params.model_savedir, "ckpt_{epoch}"),
                monitor = 'val_loss',
                save_best_only = True,
                mode = 'auto',
                save_weights_only = True,
                verbose = 0,
                save_freq = 'epoch')   
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.project.params.logs_dir)

        if self.project.params.loss == "focal":
            if self.project.params.num_classes == 2:
                self.loss = BinaryFocalLoss(gamma=2)
            else: 
                self.loss = SparseCategoricalFocalLoss(gamma=2)
            print("Using focal loss...\n")
        elif self.project.params.loss == "cross_entropy":
            self.loss = 'sparse_categorical_crossentropy'
            print("Using categorical cross entropy loss...\n")
        #elif self.project.params.loss == "polyfocal":
        #    self.loss = poly1_focal_cross_entropy
        #    print("Using poly1 focal loss...\n")
        #elif self.project.params.loss == "poly":
        #    self.loss = poly1_cross_entropy
        #    print("Using poly cross entropy loss")

    
    def conv_block(self, x, num_filters=None, kernel_size=None, strides=None):
        
        outputs = TimeDistributed(Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same'))(x)
        outputs = TimeDistributed(LeakyReLU())(outputs)
        outputs = TimeDistributed(Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same'))(outputs)
        outputs = TimeDistributed(LeakyReLU())(outputs)
        #outputs = TimeDistributed(SimAM(lambd=1e-4))(outputs)
        #outputs = TimeDistributed(ChannelAttention(ratio=8))(outputs) 
        #outputs = TimeDistributed(SpatialAttention(kernel_size=7))(outputs)
        
        return outputs
 
    def cnnlstm(self):
        
        inputs = Input(shape=self.project.params.image_shape)
                
        x = self.conv_block(inputs, num_filters=64, kernel_size=3, strides=1)
        x = TimeDistributed(MaxPool2D(pool_size=2, strides=2, padding='same'))(x)

        x = self.conv_block(x, num_filters=64, kernel_size=3, strides=1)
        x = TimeDistributed(MaxPool2D(pool_size=2, strides=2, padding='same'))(x)
        
        x = self.conv_block(x, num_filters=64, kernel_size=3, strides=1)
        x = TimeDistributed(MaxPool2D(pool_size=2, strides=2, padding='same'))(x)
         
        x = self.conv_block(x, num_filters=64, kernel_size=3, strides=1)
        x = TimeDistributed(MaxPool2D(pool_size=2, strides=2, padding='same'))(x)
                     
        x = self.conv_block(x, num_filters=64, kernel_size=3, strides=1)
        x = TimeDistributed(GlobalAveragePooling2D())(x)
             
        lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(self.project.params.num_rnn_units, return_sequences=True, return_state=True, dropout=0.2))(x) 
        state_h = Concatenate()([forward_h, backward_h])
        context_vector, attention_weights = LSTMAttentionLayer(2*self.project.params.num_rnn_units)(lstm, state_h)
        
        #context_vector = Bidirectional(LSTM(self.project.params.num_rnn_units))(x)

        context_vector = Flatten()(context_vector)
        context_vector = Dropout(0.2)(context_vector) #previous 0.2
        
        #if "poly" in self.project.params.loss:
        #    logits = Dense(self.project.params.num_classes)(context_vector) #activation='softmax')(context_vector)
        #    print("No pre softmax")
        #else:
        if self.project.params.num_classes == 2:
            logits = Dense(1, activation='sigmoid')(context_vector)
        else:
            logits = Dense(self.project.params.num_classes, activation='softmax')(context_vector)

        model = tf.keras.Model(inputs=inputs, outputs=logits, name='classifier')
        
        model.summary()
        #tf.keras.utils.plot_model(model, to_file='dau_model.png', show_shapes=True)
        
        return model


    def train(self, train_dataset, val_dataset):

        # Create directory to save model weights
        if not os.path.exists(self.project.params.model_savedir):
            os.makedirs(self.project.params.model_savedir)
        
        with open(os.path.join(self.project.params.model_savedir, 'config.json'), 'w') as file:
            json.dump(self.project.params.__dict__, file)
        
        # Distribute data in multi-GPUs in training
        if self.project.params.mode == 'train':
            strategy = tf.distribute.MirroredStrategy()
            print(f"Number of devices: {strategy.num_replicas_in_sync}")

            with strategy.scope():
                self.model = self.cnnlstm()
                opt = tf.keras.optimizers.Adam(self.project.params.lr_init)
                self.model.compile(
                        optimizer=opt, 
                        loss=self.loss, 
                        metrics=['accuracy', tfa.metrics.CohenKappa(num_classes=self.project.params.num_classes, sparse_labels=True)],
                        )
                
                # Resume training from saved checkpoints
                if self.project.params.model_continue: 
                    self.model.load_weights(tf.train.latest_checkpoint(self.project.params.model_savedir))
        
        self.model.fit(
                train_dataset,
                epochs=self.project.params.num_epochs,
                verbose=1,
                validation_data=val_dataset,
                validation_freq=1,
                callbacks=[self.cp_callback, self.tensorboard_callback],
                use_multiprocessing=True,
                )

    def test(self, dataset):
        # Use single GPU in inference
        self.model = self.cnnlstm()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.project.params.lr_init), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        self.model.load_weights(tf.train.latest_checkpoint(self.project.params.model_savedir))
        self.model.trainable = False
        print('... Trained model loaded ...')

        y_trues = []
        y_scores = []
        y_preds = []

        if self.project.params.num_classes == 2:
            for idx, (x, y) in enumerate(dataset):
                print(idx)
                y_trues.extend(y.numpy())
                y_score = self.model.predict(x)
                y_scores.extend(y_score)
                y_preds.extend(y_score>0.5)

        else: 
            for idx, (x, y) in enumerate(dataset):
                print(idx)
                if "poly" in self.project.params.loss:
                    y_trues.extend(np.argmax(y.numpy(), axis=1))
                else:
                    y_trues.extend(y.numpy())
                y_score = self.model.predict(x)
                y_scores.extend(y_score)
                y_preds.extend(np.argmax(y_score, axis=1)) 
                #print(np.argmax(np.squeeze(y_scores), axis=1))
        results = {
                'y_trues': y_trues,
                'y_preds': y_preds,
                'y_scores': y_scores
                }
        #m = tf.keras.metrics.Accuracy()
        #m.update_state(y_trues, y_preds)
        #print(m.result().numpy())
        #
        #n = tfa.metrics.CohenKappa(num_classes=3, sparse_labels=True)
        #n.update_state(y_trues, y_preds)
        #print(n.result().numpy())

        if self.project.params.mode == 'test_subjectwise':
            #sio.savemat(os.path.join('../Results/', "200813_10s.mat"), results)
            print('Saved results ...')
            fig, axs = plt.subplots(2,1)
            axs[0].step(np.arange(len(y_trues)), y_trues)
            axs[0].set_yticks([0,1,2])
            axs[0].set_yticklabels(["Wake", "NREM", "REM"])
            axs[1].step(np.arange(len(y_preds)), y_preds, color='orange')
            axs[1].set_yticks([0,1,2])
            axs[1].set_yticklabels(["Wake", "NREM", "REM"])
            plt.show()
        else:
            #sio.savemat(os.path.join('../Results', self.project.params.dataset[11:] + f"_class{self.project.params.num_classes}_test.mat"), results)
            print('Saved results ...')
        calculate_metrics(y_trues, y_preds, self.project.params.num_classes)
    
    def compute_gradcam(self, dataset):
        self.model = self.cnnlstm()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.project.params.lr_init), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        self.model.load_weights(tf.train.latest_checkpoint(self.project.params.model_savedir))
        self.model.trainable = False
        
        #gradcam(self.model, "time_distributed_23", dataset, self.project.params.gradcam_label, 10)
        for gradcam_label in range(3):
            print(f"Running GradCAM for label {gradcam_label}")
            gradcam(self.model, "time_distributed_23", dataset, gradcam_label, 10)

    def visualize_lstm_attention(self, dataset):
        
        self.model = self.cnnlstm()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.project.params.lr_init), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        self.model.load_weights(tf.train.latest_checkpoint(self.project.params.model_savedir))
        self.model.trainable = False

        visualize_lstm_attention_weights(self.model, "lstm_attention_layer", dataset, self.project.params.gradcam_label, 50)



