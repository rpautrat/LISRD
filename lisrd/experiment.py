import warnings
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import os
import sys
import argparse
import logging
import yaml
import numpy as np
import torch

from .utils.stdout_capturing import capture_outputs
from .datasets import get_dataset
from .models import get_model
from .models.base_model import Mode


logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


def _train(config, exper_dir, args):
    with open(os.path.join(exper_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    checkpoint_dir = os.path.join(exper_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    runs_dir = os.path.join(exper_dir, 'runs')
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    resume_training = args.resume_training_from
    if (resume_training != '') and (not os.path.exists(resume_training)):
        sys.exit(resume_training + ' not found.')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(
        config['data']['name'])(config['data'], device)
    net = get_model(config['model']['name'])(dataset, config['model'], device)

    n_iter = config.get('n_iter', np.inf)
    n_epoch = config.get('n_epoch', np.inf)
    assert not (np.isinf(n_iter) and np.isinf(n_epoch))
    try:
        net.train(n_iter, n_epoch, exper_dir,
                  validation_interval=config.get('validation_interval', 100),
                  save_interval=config.get('save_interval', 500),
                  resume_training=resume_training,
                  device=device)
    except KeyboardInterrupt:
        logging.info('Got Keyboard Interrupt, saving model and closing.')
    net.save(exper_dir)


def _test(config, exper_dir, args):
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        sys.exit(checkpoint_path + ' not found.')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(
        config['data']['name'])(config['data'], device)
    net = get_model(config['model']['name'])(dataset, config['model'], device)
    net.test(exper_dir, checkpoint_path, device)


def _export(config, exper_dir, args):
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        sys.exit(checkpoint_path + ' not found.')

    exper_path, experiment_name = os.path.split(exper_dir.rstrip('/'))
    export_name = args.export_name if args.export_name else experiment_name
    output_dir = os.path.join(exper_path, 'outputs', export_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(
        config['data']['name'])(config['data'], device)
    net = get_model(config['model']['name'])(dataset, config['model'], device)
    net.export(exper_dir, checkpoint_path, output_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_path', type=str,
                         help="Path to the directory of your experiment.")
    p_train.add_argument('--resume_training_from', type=str, default='',
                         help="path to a checkpoint to resume training.")
    p_train.set_defaults(func=_train)

    # Testing command
    p_test = subparsers.add_parser('test')
    p_test.add_argument('config', type=str)
    p_test.add_argument('exper_path', type=str,
                         help="Path to the directory of your experiment.")
    p_test.add_argument('checkpoint', type=str,
                         help="path to the checkpoint of the model.")
    p_test.set_defaults(func=_test)

    # Command to export the descriptors
    p_export = subparsers.add_parser('export')
    p_export.add_argument('config', type=str)
    p_export.add_argument('exper_path', type=str,
                          help="Path to the directory of your experiment.")
    p_export.add_argument('checkpoint', type=str,
                          help="path to the checkpoint of the model.")
    p_export.add_argument('--export_name', type=str,
                          help="Output directory name.")
    p_export.set_defaults(func=_export)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.full_load(f)
    exper_dir = os.path.expanduser(args.exper_path)
    if not os.path.exists(exper_dir):
        os.mkdir(exper_dir)

    logfile = 'log' if args.command == 'train' else 'log_test'
    with capture_outputs(os.path.join(exper_dir, logfile)):
        logging.info('Running command {}'.format(args.command.upper()))
        args.func(config, exper_dir, args)