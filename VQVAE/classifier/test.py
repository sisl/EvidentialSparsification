# Code adapted from: https://github.com/huyvnphan/PyTorch_CIFAR10

import os, shutil
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from module import MiniImagenet_Generated_Module

def main(hparams):
    seed_everything(0)    
    # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
    if type(hparams.gpus) == str:
        if len(hparams.gpus) == 2: # GPU number and comma e.g. '0,' or '1,'
            torch.cuda.set_device(int(hparams.gpus[0]))
    
    if hparams.dataset == 'miniimagenet':
        model = MiniImagenet_Generated_Module(hparams, pretrained=True)
    elif hparams.dataset == 'miniimagenetgenerated':
        model = MiniImagenet_Generated_Module(hparams, pretrained=True)
    
    trainer = Trainer(gpus=hparams.gpus, default_save_path=os.path.join(os.getcwd(), 'classifier/test_temp'))    
    trainer.test(model)
    shutil.rmtree(os.path.join(os.getcwd(), 'classifier/test_temp'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--gpus', default='0,')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    args = parser.parse_args()
    main(args)