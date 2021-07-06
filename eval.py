from contextlib import redirect_stderr
import os
import numpy as np
from models.model import BERTPunctuator
from dataset.dataset import SPGISpeechDataset, collate_fun, PreProcess
import argparse
from utils.model_utils import save_checkpoint, load_checkpoint, average_models
from torch.utils.data import DataLoader  
import torch
from torch import optim
import logging
from sklearn.metrics import classification_report
from transformers import BertTokenizer
from dataset import BuildDataset, collate_fun, PreProcess
from bin.executor import Executor

VOCAB = ('<PAD>', 'O', 'COMMA', 'PERIOD', 'QUESTION')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

model_dir = 'experiments/'
test_data_path = '/home/krishna/Krishna/Speech/punctuation-prediction/iwslt_data/test2011.txt'
average_models(model_dir,dst_model='experiments/final.pt')
model = BERTPunctuator(vocab_size=5).cuda()
model.load_state_dict(torch.load('experiments/final.pt'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
test_dataset = BuildDataset['iwslt'](test_data_path)

use_gpu=True
use_cuda = use_gpu >= 0 and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, collate_fn = collate_fun)
executor  = Executor()
precision, recall, f1 = executor.evaluation(model, test_loader, device)