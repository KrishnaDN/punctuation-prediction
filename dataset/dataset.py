import os
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer
import inflect
import re
import torch


VOCAB = ('<PAD>', 'O', ',', '.', '?')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class PreProcess:
    def __init__(self, train_file, test_file):
        self.train_data = [line.rstrip('\n').split('|')[-1].lower() for line in open(train_file)][1:]
        self.test_data = [line.rstrip('\n').split('|')[-1].lower() for line in open(test_file)][1:]
        self.n2w = inflect.engine()


    def convert_nums(self,text):
        words = []
        for word in text.split(' '):
            if any(i.isdigit() for i in word):
                converted = self.n2w.number_to_words(word)
                words.append(converted)
            else:
                words.append(word)
        return ' '.join(words)


    def clean_text(self, text):
        if '--' in text:
            cleaned = ' '.join(list(filter(None,text.replace('--','').split(' '))))
            return cleaned
        else:
            return text

    def final_clean(self, text):
        words = text.split(' ')
        new_words = []
        for i in range(len(words)):
            if words[i]==',' or words[i]=='.' or words[i]=='?':
                new_words.append(words[i-1]+words[i])
            else:
                new_words.append(words[i])
        return ' '.join(new_words)

    def _create_labels(self,text):
        puncts = [',','.','?']
        labels=[]
        for word in text.split(' '):
            for punc in puncts:
                try:
                    check  = word.index(punc)
                    break
                except:    
                    check=None
                    continue
            if check!=None :
                word_label = word[check]
            else:
                word_label = 'O'
            labels.append(word_label)

        if len(text.split(' ')) !=len(labels):
            print('Error')
        return (text,labels)



    def __call__(self,):
        converted_train_data = list(map(lambda x: self.convert_nums(x), self.train_data))
        converted_test_data = list(map(lambda x: self.convert_nums(x), self.test_data))
        cleaned_train_data  = list(map(lambda x: self.clean_text(x), converted_test_data))
        cleaned_test_data  = list(map(lambda x: self.clean_text(x), converted_test_data))
        
        final_cleaned_train_data  = list(map(lambda x: self.final_clean(x), cleaned_train_data))
        final_cleaned_test_data  = list(map(lambda x: self.final_clean(x), cleaned_test_data))
        
        final_train_data = list(map(lambda x: self._create_labels(x), final_cleaned_train_data))
        final_test_data = list(map(lambda x: self._create_labels(x), final_cleaned_test_data))
        
        return final_train_data, final_test_data
        

class SPGISpeechDataset:
    def __init__(self, dataset,):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sents, self.tags_li = [], []
        for item in dataset:
            text = item[0]
            cleaned_text = text.replace('.','').replace(',','').replace('?','')
            words = cleaned_text.split(' ')
            tags = item[1]
            self.sents.append(["[CLS]"] + words + ["[SEP]"])
            self.tags_li.append(["<PAD>"] + tags + ["<PAD>"])
        

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
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

def collate_fun(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)
    return words, torch.LongTensor(x), is_heads, tags, torch.LongTensor(y), seqlens


if __name__=='__main__':
    train_file = '/media/newhd/Punctuation/train'
    test_file = '/media/newhd/Punctuation/test'
    preproc = PreProcess(train_file, test_file)
    train_data, test_data = preproc()

    train_dataset = SPGISpeechDataset(train_data)
    from torch.utils import data
    train_iter = data.DataLoader(dataset=train_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=1,
                                    collate_fn=collate_fun)

    for i in range(len(train_dataset)):
        out = train_dataset.__getitem__(i)

    
    for i, batch in enumerate(train_iter):
        continue
