
import logging
from math import log
from librosa.core import audio
from dataset.iwslt_dataset import idx2tag,tag2idx
import numpy as np
import os
import torch
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score,classification_report
import tqdm

class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, scheduler, optimizer, data_loader, device, writer,):
        ''' Train one epoch
        '''
        model.train()
        clip =  50.0
        log_interval =  10
        accum_grad = 1
        rank=0
        logging.info('using accumulate grad, new batch size is {} times'
                     'larger than before'.format(accum_grad))
        num_seen_utts = 0
        num_total_batch = len(data_loader)
        for batch_idx, (words, inputs, is_heads, tags, targets, seqlens) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            num_utts = targets.size(0)
            if num_utts == 0:
                continue
            
            loss, predictions = model(inputs, targets)
            loss = loss / accum_grad
            loss.backward()

            num_seen_utts += num_utts
            if batch_idx % accum_grad == 0:
                if rank == 0 and writer is not None:
                    writer.add_scalar('train_loss', loss, self.step)
                grad_norm = clip_grad_norm_(model.parameters(), clip)
                if torch.isfinite(torch.Tensor([grad_norm])):
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                self.step += 1
            if batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                print('TRAIN Batch {}/{} loss {:.6f} lr {:.8f} rank {}'.format(
                                  batch_idx, num_total_batch,
                                  loss.item(), lr, rank))

    def validation(self, model, data_loader, device):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = 10
        num_seen_utts = 0
        total_loss = 0.0
        num_total_batch = len(data_loader)
        with torch.no_grad():
            for batch_idx, (words, inputs, is_heads, tags, targets, seqlens) in enumerate(data_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                num_utts = targets.size(0)
                if num_utts == 0:
                    continue
                loss, predictions = model(inputs, targets)

                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    print('CV Batch {}/{} loss {:.6f} history loss {:.6f}'.format(
                                      batch_idx, num_total_batch, loss.item(),
                                      total_loss / num_seen_utts))
        
        return total_loss, num_seen_utts
        

    
    def evaluation(self, model, data_loader, device):
        model.eval()
        Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
        with torch.no_grad():
            for i, (words, inputs, is_heads, tags, targets, seqlens) in enumerate(data_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model.inference(inputs)
                Words.extend(words)
                Is_heads.extend(is_heads)
                Tags.extend(tags)
                Y.extend(targets.cpu().numpy().tolist())
                Y_hat.extend(predictions.cpu().numpy().tolist())

        ## gets results and save
        with open("temp", 'w') as fout:
            for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
                y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
                preds = [idx2tag[hat] for hat in y_hat]
                try:
                    assert len(preds)==len(words.split())==len(tags.split())
                except:
                    continue 
                for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                    fout.write(f"{w} {t} {p}\n")
                fout.write("\n")
        
        ## calc metric
        target_names = ['NP','COMMA','FS','QM']
        y_true =  np.array([tag2idx[line.split()[1]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
        y_pred =  np.array([tag2idx[line.split()[2]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
        
        num_proposed = len(y_pred[y_pred>1])
        num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
        num_gold = len(y_true[y_true>1])

        print(f"num_proposed:{num_proposed}")
        print(f"num_correct:{num_correct}")
        print(f"num_gold:{num_gold}")
        try:
            precision = num_correct / num_proposed
        except ZeroDivisionError:
            precision = 1.0

        try:
            recall = num_correct / num_gold
        except ZeroDivisionError:
            recall = 1.0

        try:
            f1 = 2*precision*recall / (precision + recall)
        except ZeroDivisionError:
            if precision*recall==0:
                f1=1.0
            else:
                f1=0
        os.remove("temp")

        print("precision=%.2f"%precision)
        print("recall=%.2f"%recall)
        print("f1=%.2f"%f1)
        return precision, recall, f1