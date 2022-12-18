import argparse
import logging
import os
import sys
import warnings
from argparse import Namespace

import torch

from pipeline.test01 import Test

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def parse_args() -> Namespace:
    # args parser
    parser = argparse.ArgumentParser()

    # universal opt
    parser.add_argument('--name', default='swin', help='train process identifier')
    parser.add_argument('--folder', default='../datasets/CROPS', help='data root path')
    parser.add_argument('--val_fold', type=int, default=0, help='val fold')
    parser.add_argument('--size', default=384, help='resize image to the specified size')
    parser.add_argument('--cache', default='runs/test', help='weights cache folder')
    
    # model opt
    parser.add_argument('--model_name', default='swin_base_patch4_window12_384_in22k', help='timm model name')
    parser.add_argument('--checkpoint', default=None, help='checkpoint')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pretrained model of timm')
    parser.add_argument('--num_classes', default=33, help='num of classes')
    
    parser.add_argument('--smoothing', type=int, default=0.1, help='label smoothing')

    # dataloader opt
    parser.add_argument('--batch_size', type=int, default=32, help='dataloader batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers number')

    return parser.parse_args()

if __name__ == '__main__':
    config = parse_args()
    logging.basicConfig(level='INFO')

    # env INFO
    python_v = sys.version.split()[0]
    pytorch_v = torch.__version__
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'
    logging.info(f'python: {python_v} | pytorch: {pytorch_v} | gpu: {device_name if torch.cuda.is_available() else False}')

    # run train.
    test_process = Test(config, device)
    test_process.run()
