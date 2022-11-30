"""
Implementation of PolyLoss
Author: Xiaohui
"""

import tensorflow as tf

def poly1_cross_entropy(labels, logits):
    
    labels = tf.stop_gradient(labels)
    pt = tf.reduce_sum(labels * tf.nn.softmax(logits), axis=-1)
    CE = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    Poly1 = CE + 1.0 * (1 - pt)

    return Poly1

def poly1_focal_cross_entropy(labels, logits):

    labels = tf.stop_gradient(labels)
    pt = tf.reduce_sum(labels * tf.nn.softmax(logits), axis=-1)
    CE = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    focal_modulation = (1 - pt) ** 2.0
    FL = focal_modulation * CE
    Poly1 = FL + 1.0 * (tf.math.pow(1 - pt, 3.0))

    return Poly1
    
