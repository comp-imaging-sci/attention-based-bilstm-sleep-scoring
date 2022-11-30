import os
import argparse
import scipy.io as sio
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import matplotlib.cm as cm
from tensorflow import keras

from skimage.transform import resize
import matplotlib.pyplot as plt

def visualize_lstm_attention_weights(model, attention_layer_name, dataset, sleep_state, num_maps_to_save=10):

        # load model
        attention_model = tf.keras.Model(model.inputs, [model.get_layer(attention_layer_name).output, model.output])
        
        n = 0
        attention_recordwise = []

        for idx, (x, y) in enumerate(dataset):
            attention_outputs, preds = attention_model(x)
            top_pred_index = tf.argmax(preds[0, ])
                
            #if top_pred_index.numpy() == sleep_state and top_pred_index.numpy() == y:
            
            attention_weights = attention_outputs[1].numpy()
            attention_weights = np.squeeze(attention_weights)
            attention_recordwise.append(attention_weights)
            #attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min()) 

            #mean_raw_signal = np.mean(x[0,], axis=(1,2))
            #mean_raw_signal = (mean_raw_signal - mean_raw_signal.min()) / (mean_raw_signal.max() - mean_raw_signal.min())
            
            #fig, ax_f = plt.subplots()
            #ax_c = ax_f.twinx()
            #ax_f.plot(mean_raw_signal, color='red')
            #ax_f.set_ylabel('Global Average Signal', color='red')
            #ax_f.tick_params('y', colors='red')
            #ax_f.set_xlabel('Time points')
            #ax_c.bar(np.arange(168)+1, attention_weights, color='black')
            #ax_c.set_ylabel('Normalized Attention Weights', color='black')
            #ax_c.tick_params('y', colors='black')
            
            #fig, axs = plt.subplots(2,1)
            #axs[0].plot(np.mean(x[0,], axis=(1,2)), color='blue')
            #axs[0].set_ylabel = 'Signal Intensity'
            #axs[1].plot(np.squeeze(attention_weights), color='orange')
            #axs[1].set_xlabel = 'Index of image frame'
            #axs[1].set_ylabel = 'Attention weights'
            #plt.show()
            
            n = n+1
            print(n)
        
        #plt.plot(attention_recordwise)
        #plt.show()
        sio.savemat(f"../Results/200813_attention_weights.mat", {"att_weights": attention_recordwise})
            #if n == num_maps_to_save:    
                #    #sio.savemat(f"../Results/gradcam/{self.project.params.test_mice}_class{self.project.params.num_classes}_label{self.project.params.gradcam_label}_xiaodan.mat", {"gradcam": gradcams})
            #        print("Attention weights saved...")
            #        break


 
