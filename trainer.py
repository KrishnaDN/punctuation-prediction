import os
import numpy as np
from models.model import BERTPunctuator
from dataset import BuildDataset, collate_fun, PreProcess
import argparse
from bin.executor import Executor
from bin import BuildOptimizer, BuildScheduler
from utils.model_utils import save_checkpoint, load_checkpoint, average_models
from torch.utils.data import DataLoader  
import torch
from torch import optim
import logging

def arg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-train_filepath',type=str,default='/home/krishna/Krishna/Speech/punctuation-prediction/iwslt_data/train2012.txt')
    parser.add_argument('-test_filepath',type=str, default='/home/krishna/Krishna/Speech/punctuation-prediction/iwslt_data/dev2012.txt')
    parser.add_argument('-dataset',type=str, default='iwslt')
    
    parser.add_argument('-model_dir',type=str, default='experiments')
    
    parser.add_argument('-num_classes', action="store_true", default=5)
    parser.add_argument('-batch_size', action="store_true", default=32)
    parser.add_argument('-use_gpu', action="store_true", default=True)
    parser.add_argument('-num_epochs', action="store_true", default=100)
    args = parser.parse_args()
    return args



def main():
    args = arg_parser()
    if args.dataset=='spgispeech':
        preprocess = PreProcess(args.train_filepath, args.test_filepath)
        train_data, test_data = preprocess()
        train_dataset = BuildDataset['spgispeech'](train_data)
        test_dataset = BuildDataset['spgispeech'](test_data)
    else:
        train_dataset = BuildDataset[args.dataset](args.train_filepath)
        test_dataset = BuildDataset[args.dataset](args.test_filepath)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn = collate_fun)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn = collate_fun)
    
    model = BERTPunctuator(vocab_size=args.num_classes)
    use_cuda = args.use_gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model_dir = args.model_dir
    writer = None
    executor = Executor()
    optimizer = BuildOptimizer['adam'](model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = BuildScheduler['constant'](optimizer, lr=0.0001)
    infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    if start_epoch == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)
        
    for epoch in range(start_epoch, args.num_epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, scheduler,optimizer, train_loader, device,
                    writer)
        
        precision, recall, f1 = executor.evaluation(model, test_loader, device)
        
        print('Epoch {} Test info Precision {}  Recall {} F1-score {}'.format(epoch, precision, recall, f1))
        
        save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
        save_checkpoint(
            model, save_model_path, {
                'epoch': epoch,
                'lr': lr,
                'step': executor.step,
                'precision': float(precision),
                'recall': float(recall),
                'f1-score': float(f1),
            })


    
    
if __name__=='__main__':
    main()