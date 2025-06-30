import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
import random


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        attn = self.attn(self.norm(x))
        # print(attn.size())
        x = x + attn

        return x


class ViT(nn.Module):
    def __init__(self, n_classes, top_k=1, n_layers=2, args=None):
        super(ViT, self).__init__()
        
        input_dim = args.input_dim
        emb_dim = args.emb_dim
        
        self._fc1 = nn.Sequential(nn.Linear(input_dim, emb_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, emb_dim))
        self.n_classes = n_classes
        
        trans_layers = [TransLayer(dim=emb_dim) for _ in range(args.n_layers)]
        self.layer = nn.Sequential(*trans_layers)
        
        self.norm = nn.LayerNorm(emb_dim)
        self._fc2 = nn.Linear(emb_dim, self.n_classes)
        self._fc3 = nn.Linear(emb_dim, self.n_classes)
        
        self.args = args
        
        
    def forward(self, x_path, x_omic=None):
        h = x_path
        # print(h.size())
        h = self._fc1(h) 
        
        #---->cls_token
        cls_tokens = self.cls_token.cuda()
        h = torch.cat((cls_tokens, h), dim=0)

        #---->Translayer 
        h = torch.unsqueeze(h, dim=0)
        h = self.layer(h)
        h = torch.squeeze(h, dim=0)
        # print(h.size())
        
        #---->cls_token
        cls_token_output = self.norm(h)[0]
        y_outputs = self.norm(h)[1:]

        #---->predict
        cls_logit = self._fc2(cls_token_output) #[n_classes]
        Y_hat = torch.argmax(cls_logit, dim=0).reshape(1, 1)
        Y_prob = F.softmax(cls_logit, dim=0).unsqueeze(dim=0)

        patch_logits = self._fc3(y_outputs)
        patch_probs = F.softmax(patch_logits, dim=0)
        
        result_dict = {}
        
        result_dict.update({'all_instance': patch_logits})
        hazards = torch.sigmoid(torch.unsqueeze(cls_logit, dim=0))
        S = torch.cumprod(1 - hazards, dim=1)

        
        return hazards, S, Y_hat