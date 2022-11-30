import argparse
import tensorflow as tf
import importlib
import os, random
import json
import sys
import numpy as np

class project_params:
    def __init__(self, params):
        self.params = params

def main(project):
    dataloader = selected_dataloader.dataloader_sleep_tfrecords(project)
    train_dataset, val_dataset, test_dataset = dataloader.load_data()
    model = selected_model.model(project)

    if project.params.mode == 'train':
        model.train(train_dataset, val_dataset)
    elif project.params.mode == 'gradcam':
        model.compute_gradcam(test_dataset)
    elif project.params.mode == 'attention_weights':
        model.visualize_lstm_attention(test_dataset)
    else:
        model.test(test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model', type=str, default='model_cnnlstm')
    parser.add_argument('--dataset', type=str, default='dataloader_sleep_tfrecords_local')
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    

    args = parser.parse_known_args()[0]
    if args.seed >= 0:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    selected_model = importlib.import_module(args.model)
    parser = selected_model.add_arguments(parser)

    selected_dataloader = importlib.import_module(args.dataset)
    parser = selected_dataloader.add_arguments(parser)
    args = parser.parse_args()
    project = project_params(args)
    main(project)


