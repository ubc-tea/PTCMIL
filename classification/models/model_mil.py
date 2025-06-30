import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

class Maxpool(nn.Module):
    def __init__(self, gate = True, dropout = False, n_classes = 2, args=None):
        super(Maxpool, self).__init__()
        # assert n_classes == 2
        
        size = [args.input_dim, 384]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        fc.append(nn.Linear(size[1], n_classes))
        self.classifier= nn.Sequential(*fc)
        initialize_weights(self)

    def forward(self, h, return_features=False):
        if return_features:
            h = self.classifier.module[:3](h)
            logits = self.classifier.module[3](h)
        else:
            logits  = self.classifier(h) # K x 1
        
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], 1, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim=1)[1]
        Y_prob = F.softmax(top_instance, dim=1) 
        
        results_dict = {}

        return top_instance, Y_prob, Y_hat, y_probs, results_dict

    
class Meanpool(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, n_classes = 2, top_k=1, args=None):
        super(Meanpool, self).__init__()
        # assert n_classes == 2
        
        size = [args.input_dim, 384]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))

        fc.append(nn.Linear(size[1], n_classes))
        self.classifier= nn.Sequential(*fc)
        initialize_weights(self)
        self.top_k=top_k

    def forward(self, h, return_features=False):
        if return_features:
            h = self.classifier.module[:3](h)
            logits = self.classifier.module[3](h)
        else:
            logits  = self.classifier(h) # K x 1
        
        y_probs = F.softmax(logits, dim = 1)
        
        mean_logits = logits.mean(dim=0).unsqueeze(0)
        
        Y_hat = torch.topk(mean_logits, 1, dim=1)[1]
        Y_prob = F.softmax(mean_logits, dim=1) 
        
        results_dict = {}
        
        return mean_logits, Y_prob, Y_hat, y_probs, results_dict

