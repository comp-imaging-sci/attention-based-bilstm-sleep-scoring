import re, glob, os
import tensorflow as tf
from create_tfrecords import parse_tfrecord_fn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, f1_score
import seaborn as sns

import scipy.io as sio
def calculate_metrics(y_trues, y_preds, num_classes):
    """
    Function to get classification report
    """
    if num_classes == 3:
        states = ['Wakefulness', 'NREM', 'REM']   
    elif num_classes == 2:
        states = ['Wakefulness', 'Sleep']
    cm = confusion_matrix(y_trues, y_preds)
    norm_cm = confusion_matrix(y_trues, y_preds, normalize='true')
    group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in norm_cm.flatten()]
    labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(num_classes,num_classes)
   
    print(classification_report(y_trues, y_preds, target_names=states))           
    print(f"Cohen Kappa Score: {cohen_kappa_score(y_trues, y_preds)}")    #, weights='linear'
    print(f"Unweighted F1-score: {f1_score(y_trues, y_preds, average='micro')}")
    print(f"Weighted F1-score: {f1_score(y_trues, y_preds, average='weighted')}")
    sns.set(font_scale=1.5)
    s = sns.heatmap(
            cm,
            cmap="Greens",
            annot=labels,
            cbar_kws={"orientation": "vertical", "label": "number of epochs"},
            xticklabels=states,
            yticklabels=states,
            fmt='',
            #annot_kws={"fontsize":14}
            )
    s.set_ylabel('True Labels')
    s.set_xlabel('Predictions')
    plt.show()


def plot_hynogram_tfrecords(data_path, mouse_name):
    """
    Function to plot hypnogram on tfrecords
    """
    def _get_regexp(fname):
         
        m = re.match(r"(\d+)-(\w+\d+)-fc(\d+)-GSR_Ben_epoch(\d+).tfrecords", os.path.basename(fname))
        name_parts = (m.groups())
        return int(name_parts[2]), int(name_parts[3])

    fc_fnames = sorted(glob.glob(os.path.join(data_path, f"{str(mouse_name)}*")), key=lambda x: _get_regexp(x))
    labels = []

    for fc_fname in fc_fnames:
        print(os.path.basename(fc_fname))
        dataset = tf.data.TFRecordDataset(fc_fname).map(map_func=parse_tfrecord_fn)
        for _, y in dataset:
            labels.append(y.numpy())
    
    #wake = np.ma.masked_where(labels == 0, labels)
    #nrem = np.ma.masked_where(labels == 1, labels)
    #rem = np.ma.masked_where(labels == 2, labels)
    print(labels)
    sio.savemat('../Results/test.mat', {"raw": labels})
    plt.plot(labels, drawstyle='steps-mid')
    plt.show()

def count_sleep_state(data_path, mouse_name):
    """
    Function to count total number of each state
    """
    def _get_regexp(fname):
         
        m = re.match(r"(\d+)-(\w+\d+)-fc(\d+)-GSR_Ben_epoch(\d+).tfrecords", os.path.basename(fname))
        name_parts = (m.groups())
        return int(name_parts[2]), int(name_parts[3])

    fc_fnames = sorted(glob.glob(os.path.join(data_path, f"{str(mouse_name)}*")), key=lambda x: _get_regexp(x))
    labels = []

    for fc_fname in fc_fnames:
        print(os.path.basename(fc_fname))
        dataset = tf.data.TFRecordDataset(fc_fname).map(map_func=parse_tfrecord_fn)
        for _, y in dataset:
            labels.append(y.numpy())
    labels = np.array(labels)
    print(f"Wake: {np.count_nonzero(labels==0)} epochs, NREM: {np.count_nonzero(labels==1)} epochs, REM: {np.count_nonzero(labels==2)} epochs")

def compare_two_human_rater(y_trues1, y_trues2):
    """
    Compare scoring by Eric and Nick
    """
    states = ['Wakefulness', 'NREM', 'REM']   
    print(classification_report(y_trues1, y_trues2, target_names=states))           
    print(f"Cohen Kappa Score: {cohen_kappa_score(y_trues1, y_trues2)}")    #, weights='linear'
    print(f"Weighted F1-score: {f1_score(y_trues1, y_trues2, average='weighted')}")

if __name__ == "__main__":
    data_path = '/shared/planck/MRI/xiaohui/mouse_optical/sleep_stage/2022-Ben-10s-raw-masked-tempzscore-broadband-states-tfrecords'
    mouse_name = "200128-L2" 
    #plot_hynogram_tfrecords(data_path, mouse_name)
    #count_sleep_state(data_path, mouse_name)

    y_trues = sio.loadmat('../Results/nick_eric_scoring.mat')
    compare_two_human_rater(y_trues['y_trues_eric'], y_trues['y_trues_nick'])





