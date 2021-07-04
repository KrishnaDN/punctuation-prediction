import torch
import torch.nn as nn
from transformers import BertModel

class BERTPunctuator(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)


    def forward(self, inputs, targets, ):
        encoded = self.bert(inputs)
        logits = self.fc(encoded[0])
        pred_labels = logits.argmax(-1)
        loss = self.compute_loss(logits, targets)
        return loss, pred_labels

    def compute_loss(self, logits, targets):
        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        targets = targets.view(-1)  # (N*T,)
        loss = self.criterion(logits,targets )
        return loss


    def inference(self, x):
        self.bert.eval()
        encoded = self.bert(x)
        logits = self.fc(encoded[0])
        y_hat = logits.argmax(-1)
        return y_hat

    