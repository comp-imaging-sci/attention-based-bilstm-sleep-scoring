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

def gradcam(model, last_conv_layer_name, dataset, sleep_state, num_maps_to_save=10):
    """
    Grad-CAM
    """
    # load model
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    
    n = 0
    gradcam_maps = []

    for idx, (x, y) in enumerate(dataset):
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(x)
            top_pred_index = tf.argmax(preds[0, ])
            #print(preds)
            #print(top_pred_index)
            print(tf.math.reduce_max(preds[0,]))
            top_class_channel = preds[:, top_pred_index]
            
        if top_pred_index.numpy() == sleep_state and top_pred_index.numpy() == y:
        
            grads = tape.gradient(top_class_channel, last_conv_layer_output)
            last_conv_layer_output = last_conv_layer_output.numpy()[0]
            
            # Guided grad
            #gate_f = tf.cast(last_conv_layer_output > 0, 'float32')
            #gate_r = tf.cast(grads > 0, 'float32')
            #guided_grads = gate_f*gate_r*grads
             
            pooled_grads = tf.reduce_mean(grads, axis=(0,2,3))
            pooled_grads = pooled_grads.numpy()
            
            heatmap = np.zeros(last_conv_layer_output.shape[0:3])
            for tp in range(pooled_grads.shape[0]):
                for i in range(pooled_grads.shape[-1]):
                    heatmap[tp,:,:] += last_conv_layer_output[tp,:,:,i] * pooled_grads[tp,i]
            
            heatmap = np.maximum(heatmap, 0)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + K.epsilon())
            #heatmap = resize(heatmap, (168, 128, 128))
            heatmap = resize(heatmap, np.shape(np.squeeze(x.numpy())))
            
            #for tp in range(168):
            demo_map = heatmap
            #plt.imshow(x[0, 1, :, :], cmap='gray')
            #plt.imshow(np.mean(demo_map, axis=0), cmap='jet', alpha=0.5)
            #plt.axis('off')
            #plt.show()
            gradcam_maps.append(heatmap)
            
            n = n+1
            print(n)
    
    sio.savemat(f"../Results/gradcam/201125_avg_hemis_label{sleep_state}.mat", {"subject_avg_gradcam": np.mean(np.array(gradcam_maps), axis=(0,1)), "example_data": np.squeeze(x.numpy())})
            #plt.show()

            #if n == num_maps_to_save:  
            #    #sio.savemat(f"../Results/gradcam/{self.project.params.test_mice}_class{self.project.params.num_classes}_label{self.project.params.gradcam_label}_xiaodan.mat", {"gradcam": gradcams})
            #    print("Grad-CAM saved...")
            #    break

def gradcam_plus(model, last_conv_layer_name, dataset, sleep_state, num_maps_to_save=10):
    """
    Grad-CAM++
    """
    # load model
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    
    n = 0
    gradcam_maps = []

    for idx, (x, y) in enumerate(dataset):
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    last_conv_layer_output, preds = grad_model(x)
                    top_pred_index = tf.argmax(preds[0, ])
                    #print(preds)
                    #print(top_pred_index)
                    print(tf.math.reduce_max(preds[0,]))
                    top_class_channel = preds[:, top_pred_index]
            
                    #if top_pred_index.numpy() == sleep_state and top_pred_index.numpy() == y:
        
                    conv_first_grads = tape3.gradient(top_class_channel, last_conv_layer_output)
                conv_second_grads = tape2.gradient(conv_first_grads, last_conv_layer_output)
            conv_third_grads = tape1.gradient(conv_second_grads, last_conv_layer_output)
        
        global_sum = np.sum(last_conv_layer_output, axis=(0,2,3))
        
        alpha_num = conv_second_grad[0]
        alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)
            
        alphas = alpha_num/alpha_denom
        alpha_normalization_constant = np.sum(alphas, axis=(0,1))
        alphas /= alpha_normalization_constant

        weights = np.maximum(conv_first_grad[0], 0.0)

        deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
        grad_cam_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)
                      
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + K.epsilon())
        heatmap = resize(heatmap, (168, 128, 128))
        
        #for tp in range(168):
        demo = heatmap
        plt.imshow(x[0, 1, :, :], cmap='gray')
        plt.imshow(np.mean(demo_map, axis=0), cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.show()
        #gradcam_maps.append(heatmap)
        
        n = n+1
        print(n)

        if n == num_maps_to_save:    
        #    #sio.savemat(f"../Results/gradcam/{self.project.params.test_mice}_class{self.project.params.num_classes}_label{self.project.params.gradcam_label}_xiaodan.mat", {"gradcam": gradcams})
            print("Grad-CAM saved...")
            break


