import os
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer
import inflect
import re
import torch

VOCAB = ('<PAD>', 'O', 'COMMA', 'PERIOD', 'QUESTION')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class IWSLTDataset:
    def __init__(self, dataset_path, window_len = 128):
        self.dataset_path = dataset_path
        self.window_len = window_len
        self.data = [line.rstrip('\n') for line in open(self.dataset_path)][:-1]
        self.segments = [self.data[i:i+128]   for i in range(0,len(self.data),128)]
        self.tokenizer = tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self,):
        return len(self.segments)
    
    def __getitem__(self, idx):
        words = [ item.split('\t')[0] for item in self.segments[idx]]
        words = ["[CLS]"] + words + ["[SEP]"]
        tags =  [item.split('\t')[1] for item in self.segments[idx]]
        tags = ["<PAD>"] + tags + ["<PAD>"]
        new_words = []
        new_tags = []
        for k in range(len(words)):
            if words[k]=='':
                continue
            else:
                new_words.append(words[k])
                new_tags.append(tags[k])
        words = new_words
        tags = new_tags
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
        seqlen = len(y)
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen
        

if __name__=='__main__':
    dataset_path = '/home/krishna/Krishna/Speech/punctuation-prediction/iwslt_data/train2012.txt'
    gen = IWSLTDataset(dataset_path)
    for i in range(len(gen)):
        gen.__getitem__(i)