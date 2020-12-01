import os
import sys
sys.path.insert(0,os.getcwd())

import numpy
import torch
import logging
from tqdm import tqdm
from kaldiio import WriteHelper
from importlib import import_module

from torch.utils.data import DataLoader
from util.dataset_loader import EvalDataset

def compute_tsvad_weights(writer, utt, preds):
    for i in range(4):
        pred = preds[:, i]
        uid = utt + '-' + str(i+1)
        writer(uid, pred)

def inference(infer_config):
    # Initial
    model_type = infer_config.get('model_type', 'tsvad')
    model_path = infer_config.get('model_path', '')
    output_dir = infer_config.get('output_dir', '')
    feats_dir  = infer_config.get('feats_dir', '')
    ivectors_dir = infer_config.get('ivectors_dir', '')

    # Load Model
    module = import_module('model.{}'.format(model_type))
    MODEL = getattr(module, 'Model')
    model = MODEL()
    model.load_state_dict(torch.load(model_path)['model'])
    
    print (model)

    model = model.cuda()

    # Load evaluation data
    evalset = EvalDataset(feats_dir=feats_dir, ivectors_dir=ivectors_dir)
    eval_loader = DataLoader(evalset, num_workers=0, shuffle=False, batch_size=1)

    # Prepare logger
    logger = logging.getLogger("logger")
    handler1 = logging.StreamHandler()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s",
                                  datefmt="%m-%d %H:%M:%S")
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    logger.info("Evaluation utterances: {}".format(len(evalset)))

    # ================ MAIN EVALUATION LOOP! ===================

    logger.info("Start evaluation...")
    
    model.eval()
    with WriteHelper('ark,t:{}/weights.ark'.format(output_dir)) as writer:
        for i, batch in tqdm(enumerate(eval_loader)):
            utt, _, _ = batch
            with torch.no_grad():
                preds = model.inference(batch).squeeze(0).cpu().numpy()
                compute_tsvad_weights(writer, utt[0], preds)
                
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/config_dcase.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--model_path', type=str, default=None,
                        help='model path to load')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='output directory')
    parser.add_argument('-f', '--feats_dir', type=str, default=None,
                        help='output directory')
    parser.add_argument('-i', '--ivectors_dir', type=str, default=None,
                        help='output directory')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='Using gpu #')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    infer_config = config["infer_config"]
    global model_config
    model_config = config['model_config']

    if args.model_path is not None:
        infer_config['model_path'] = args.model_path
    if args.output_dir is not None:
        infer_config['output_dir'] = args.output_dir
    if args.feats_dir is not None:
        infer_config['feats_dir'] = args.feats_dir
    if args.ivectors_dir is not None:
        infer_config['ivectors_dir'] = args.ivectors_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    inference(infer_config)
