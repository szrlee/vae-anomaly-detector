"""
Main experiment
"""
import json
import os
import argparse
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
from configparser import ConfigParser
from datetime import datetime
from scipy.special import logsumexp

from vae.vae import VAE
from utils.data import SpamDataset
from utils.feature_extractor import FeatureExtractor
from constants import MODELS
from utils.visualization import mean_confidence_interval


def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='VAE spam detector')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument(
        '--globals', type=str, default='./configs/globals.ini', 
        help="Path to the configuration file containing the global variables "
             "e.g. the paths to the data etc. See configs/globals.ini for an "
             "example."
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help="Id of the model configuration file. If this argument is not null, "
             "the system will look for the configuration file "
             "./configs/{args.model}/{args.model}{args.config}.ini"
    )
    parser.add_argument(
        '--restore_filename', type=str, required=True, 
        help="Path to a model checkpoint containing trained parameters. " 
             "If provided, the model will load the trained parameters before "
             "resuming training or making a prediction. By default, models are "
             "saved in ./checkpoints/<args.model><args.config>/<date>/"
    )
    return parser.parse_args()


def load_config(args):
    """
    Load .INI configuration files
    """
    config = ConfigParser()

    # Load global variable (e.g. paths)
    config.read(args.globals)

    # Path to the directory containing the model configurations
    model_config_dir = os.path.join(config['paths']['configs_directory'], '{}/'.format(args.model))

    # Load default model configuration
    default_model_config_filename = '{}.ini'.format(args.model)
    default_model_config_path = os.path.join(model_config_dir, default_model_config_filename)
    config.read(default_model_config_path)

    if args.config:
        model_config_filename = '{}{}.ini'.format(args.model, args.config)
        model_config_path = os.path.join(model_config_dir, model_config_filename)
        config.read(model_config_path)

    config.set('model', 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    return config

def eval(config, testloader):
    storage = {
        # 'll_precision': None, 'll_recall': None, 
        'log_densities': None, 'params': None,
        'ground_truth': None
    }
    input_dim = testloader.dataset.input_dim_
    vae = VAE(input_dim, config, checkpoint_directory=None)
    vae.to(config['model']['device'])
    if args.restore_filename is not None:
        vae.restore_model(args.restore_filename, epoch=None)
    vae.eval()
    precisions, recalls, all_log_densities = [], [], []
    # z sample sizes: 100
    for i in range(100):
        print("evaluation round {}".format(i))
        _, _, precision, recall, log_densities, ground_truth = vae.evaluate(testloader)
        precisions.append(precision)
        recalls.append(recall)
        all_log_densities.append(np.expand_dims(log_densities, axis=1))
    print(mean_confidence_interval(precisions))
    print(mean_confidence_interval(recalls))
    all_log_densities = np.concatenate(all_log_densities, axis=1)
    # log sum exponential
    storage['log_densities'] = logsumexp(all_log_densities, axis=1) - np.log(100)
    storage['ground_truth'] = ground_truth
    # storage['ll_precision'] = mean_confidence_interval(precisions)
    # storage['ll_recall'] = mean_confidence_interval(recalls)
    # storage['params'] = self._get_parameters(testloader)
    pkl_filename = './results/test/{}{}/{}.pkl'.format(config['model']['name'], \
      config['model']['config_id'], args.restore_filename)
    os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
    with open(pkl_filename, 'wb') as _f:
        pickle.dump(storage, _f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = argparser()
    config = load_config(args)

    # Get data path
    data_dir = config.get("paths", "data_directory")
    test_data_file_name = config.get("paths", "test_data_file_name")
    test_csv_path = os.path.join(data_dir, test_data_file_name)
    data_file_name = config.get("paths", "data_file_name")
    corpus_csv_path = os.path.join(data_dir, data_file_name)
    
    # Set text processing function
    transformer = FeatureExtractor(config)
    raw_documents = transformer.get_raw_documents(corpus_csv_path)
    transformer.fit(raw_documents)
    transformer.log_vocabulary('data/test_vocab.txt')

    test_data = SpamDataset(
        test_csv_path,
        label2int=json.loads(config.get("data", "label2int")),
        transform=transformer.vectorize)

    # No shuffle data in testset: to guarentee the same order of predictions
    # from differemnt models.
    testloader = DataLoader(
        test_data,
        batch_size=config.getint("training", "batch_size"),
        shuffle=False,
        num_workers=0,
        pin_memory=False)

    eval(config, testloader)
