import os
import sys
sys.path.insert(0,os.getcwd())

import time
import torch
import logging
import numpy as np
from pathlib import Path
from importlib import import_module

from torch.utils.data import DataLoader
from util.dataset_loader import Dataset


def train(train_config): 
    # Initial
    output_directory     = train_config.get('output_directory', '')
    max_iter             = train_config.get('max_iter', 100000)
    batch_size           = train_config.get('batch_size', 128)
    nframes              = train_config.get('nframes', 40)
    iters_per_checkpoint = train_config.get('iters_per_checkpoint', 10000)
    iters_per_log        = train_config.get('iters_per_log', 1000)
    seed                 = train_config.get('seed', 1234)
    checkpoint_path      = train_config.get('checkpoint_path', '')
    trainer_type         = train_config.get('trainer_type', 'basic')

    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)   

    # Initial trainer
    module = import_module('trainer.{}'.format(trainer_type), package=None)
    TRAINER = getattr( module, 'Trainer')
    trainer = TRAINER( train_config, model_config)
    try:
        collate_fn = getattr( module, 'collate')
    except:
        collate_fn = None

    # Load checkpoint if the path is given 
    iteration = 1
    if checkpoint_path != "":
        iteration = trainer.load_checkpoint( checkpoint_path)
        iteration += 1  # next iteration is iteration + 1

    # Load training data
    trainset = Dataset(train_config['training_dir'], nframes)    
    train_loader = DataLoader(trainset, num_workers=32, shuffle=True,
                              batch_size=batch_size,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)

    # Get shared output_directory ready
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Prepare logger
    logger = logging.getLogger("logger")
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=str(output_directory/'Stat'))
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s",
                                  datefmt="%m-%d %H:%M:%S")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    logger.info("Output directory: {}".format(output_directory))
    logger.info("Training utterances: {}".format(len(trainset)))
    logger.info("Batch size: {}".format(batch_size))
    logger.info("# of frames per sample: {}".format(nframes))

    # ================ MAIN TRAINNIG LOOP! ===================
    
    logger.info("Start traininig...")

    loss_log = dict()
    while iteration <= max_iter:
        for i, batch in enumerate(train_loader):
            
            iteration, loss_detail, lr = trainer.step(batch, iteration=iteration)

            # Keep Loss detail
            for key,val in loss_detail.items():
                if key not in loss_log.keys():
                    loss_log[key] = list()
                loss_log[key].append(val)
            
            # Save model per N iterations
            if iteration % iters_per_checkpoint == 0:
                checkpoint_path =  output_directory / "{}_{}".format(time.strftime("%m-%d_%H-%M", time.localtime()),iteration)
                trainer.save_checkpoint( checkpoint_path)

            # Show log per M iterations
            if iteration % iters_per_log == 0 and len(loss_log.keys()) > 0:
                mseg = 'Iter {}:'.format( iteration)
                for key,val in loss_log.items():
                    mseg += '  {}: {:.6f}'.format(key,np.mean(val))
                mseg += '  lr: {:.6f}'.format(lr)
                logger.info(mseg)
                loss_log = dict()

            if iteration > max_iter:
                break

    print('Finished')
        

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='tsvad_config.json',
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_directory', type=str, default=None,
                        help='Directory for checkpoint output')
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help='checkpoint path to keep training')
    parser.add_argument('-T', '--training_dir', type=str, default=None,
                        help='Traininig dictionary path')

    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='Using gpu #')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global model_config
    model_config = config["model_config"]

    if args.output_directory is not None:
        train_config['output_directory'] = args.output_directory
    if args.checkpoint_path is not None:
        train_config['checkpoint_path'] = args.checkpoint_path

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    train(train_config)
