# -*- coding: utf-8 -*-
"""
usage: save_ptm.py [-h] [--bert] <config> <ckpt>

Save the PreTrainedModel(s) wrapped inside the Trainee (LightningModule).

positional arguments:
  <config>    Path to the lightning config file (YAML).
  <ckpt>      Path to the lightning checkpoint.

optional arguments:
  -h, --help  show this help message and exit
  --bert      For DPR-based BiEncoder, save BertModel instead of DPR*Encoder
"""
import argparse
import yaml
from pathlib import Path

from . import trainee


def main(config, ckpt, **kwargs):
    with open(config, 'rt') as file:                                                               
        config = yaml.load(file, yaml.Loader) 
    class_name = config['model']['class_path'].split('.')[-1]
    Class = getattr(trainee, class_name)
    model = Class.load_from_checkpoint(ckpt, **config['model']['init_args'])
    ckpt_path = ckpt.with_suffix('')
    model.save_pretrained(ckpt_path, **kwargs)
    
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Save the PreTrainedModel(s) wrapped inside the Trainee (LightningModule).')
    parser.add_argument('config', metavar='<config>', type=str, help='Path to the lightning config file (YAML).')
    parser.add_argument('ckpt', metavar='<ckpt>', type=Path, help='Path to the lightning checkpoint.')
    parser.add_argument('--bert', action='store_true', help='For DPR-based BiEncoder, save BertModel instead of DPR*Encoder')
    args = parser.parse_args()
    main(**vars(args))
    