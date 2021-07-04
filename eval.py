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
VOCAB = ('<PAD>', 'O', ',', '.', '?')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class Evaluation:
    def __init__(self, model_dir, window_len = 20):
        self.model_dir = model_dir
        average_models(self.model_dir,dst_model='experiments/final.pt')
        self.model = BERTPunctuator(vocab_size=5).cuda()
        self.model.load_state_dict(torch.load('experiments/final.pt'))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.window_len = window_len



    def eval_fisher(self, fisher_data):
        import re
        read_data = open(fisher_data, 'r').read().strip().split("\n")
        sents, tags_li = [], [] # list of lists
        for entry in read_data:
            words_unproc = entry.split('\t')[0].split(' ')
            tags = entry.split('\t')[1].split(' ')
            words=[]
            for word in words_unproc:
                cleaned_string = re.sub('[^A-Za-z0-9]+', '', word)
                if cleaned_string=='':
                    continue
                words.append(cleaned_string)
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<PAD>"] + tags + ["<PAD>"])

        with torch.no_grad():
            with open("temp", 'w') as fout:
                for i in range(len(tags_li)):
                    words, tags = sents[i], tags_li[i]
                    x, y = [], [] # list of ids
                    is_heads = [] # list. 1: the token is the first piece of a word
                    for w, t in zip(words, tags):
                        tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                        xx = self.tokenizer.convert_tokens_to_ids(tokens)
                        
                        is_head = [1] + [0]*(len(tokens) - 1)
                        
                        t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
                        yy = [tag2idx[each] for each in t]  # (T,)

                        x.extend(xx)
                        is_heads.extend(is_head)
                        y.extend(yy)
                        
                    assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"
                    # seqlen
                    seqlen = len(y)
                    tags = " ".join(tags)
                    inputs = torch.LongTensor(x).unsqueeze(0)
                    predictions = self.model.inference(inputs.cuda())
                    outs = list(predictions[0].cpu().numpy()) 

                    preds = [idx2tag[hat] for hat in outs]
                    
                    for w, t, p in zip(words[1:-1], tags[1:-1], preds[1:-1]):
                        fout.write(f"{w} {t} {p}\n")
                    fout.write("\n")


    def eval_iwslt(self, test_data_path):
        from collections import OrderedDict
        self.model.eval()
        labels_dict = {'<PAD>':0,'O':1,'COMMA':2,'PERIOD':3, 'QUESTION':4}
        read_data = [line.rstrip('\n') for line in open(test_data_path)]
        with torch.no_grad():
            with open("temp", 'w') as fout:
                for i in range(0, len(read_data), self.window_len):
                    segment = read_data[i: i+self.window_len]
                    words  = ['[CLS]'] +[item.split('\t')[0] for item in segment]+ ['[SEP]']
                    tags = ['<PAD>'] +[item.split('\t')[1] for item in segment]+ ['<PAD>']
                    x,y=[],[]
                    for w, t in zip(words, tags):
                        tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                        xx = self.tokenizer.convert_tokens_to_ids(tokens)
                        is_head = [1] + [0]*(len(tokens) - 1)
                        t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
                        yy = [labels_dict[each] for each in t]  # (T,)
                        x.extend(xx)
                        y.extend(yy)
                    assert len(x)==len(y)
                    inputs = torch.LongTensor(x).unsqueeze(0)
                    predictions = self.model.inference(inputs.cuda())
                    outs = list(predictions[0].cpu().numpy()) 

                    preds = [idx2tag[hat] for hat in outs]
                    
                    for w, t, p in zip(words[1:-1], tags[1:-1], preds[1:-1]):
                        fout.write(f"{w} {t} {p}\n")
                    fout.write("\n")
                    
        ## calc metric
        
        #target_names = ['NP','COMMA','FS','QM']
        #y_true =  np.array([labels_dict[line.split()[1]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
        #y_pred =  np.array([tag2idx[line.split()[2]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
        #print(classification_report(y_true, y_pred, target_names=target_names, digits=4)) 



model_val = Evaluation(model_dir = 'experiments/', )
fisher_data = '/home/krishna/Krishna/Speech/bert_punctuation/fisher_data/test_fisher.txt'
test_data_path = '/home/krishna/Krishna/Speech/punctuation-prediction/iwslt_data/test2011.txt'
model_val.eval_fisher(fisher_data)




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
VOCAB = ('<PAD>', 'O', ',', '.', '?')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

window_len = 20
model_dir = 'experiments/'
average_models(model_dir,dst_model='experiments/final.pt')
model = BERTPunctuator(vocab_size=5).cuda()
model.load_state_dict(torch.load('experiments/final.pt'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
window_len = window_len

import re
read_data = open(fisher_data, 'r').read().strip().split("\n")
sents, tags_li = [], [] # list of lists
for entry in read_data:
    words_unproc = entry.split('\t')[0].split(' ')
    tags = entry.split('\t')[1].split(' ')
    words=[]
    for word in words_unproc:
        cleaned_string = re.sub('[^A-Za-z0-9]+', '', word)
        if cleaned_string=='':
            continue
        words.append(cleaned_string)
    sents.append(["[CLS]"] + words + ["[SEP]"])
    tags_li.append(["<PAD>"] + tags + ["<PAD>"])

with torch.no_grad():
    with open("temp", 'w') as fout:
        for i in range(len(tags_li)):
            words, tags = sents[i], tags_li[i]
            x, y = [], [] # list of ids
            is_heads = [] # list. 1: the token is the first piece of a word
            for w, t in zip(words, tags):
                tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                xx = tokenizer.convert_tokens_to_ids(tokens)
                
                is_head = [1] + [0]*(len(tokens) - 1)
                
                t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
                yy = [tag2idx[each] for each in t]  # (T,)

                x.extend(xx)
                is_heads.extend(is_head)
                y.extend(yy)
                
            assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"
            # seqlen
            seqlen = len(y)
            inputs = torch.LongTensor(x).unsqueeze(0)
            predictions = model.inference(inputs.cuda())
            outs = list(predictions[0].cpu().numpy()) 

            preds = [idx2tag[hat] for hat in outs]
            
            for w, t, p in zip(words[1:-1], tags[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

target_names = ['NP','COMMA','FS','QM']
y_true =  np.array([tag2idx[line.split()[1]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
y_pred =  np.array([tag2idx[line.split()[2]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
print(classification_report(y_true, y_pred, target_names=target_names, digits=4)) 