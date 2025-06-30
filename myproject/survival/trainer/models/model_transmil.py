import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,   
            pinv_iterations = 6,    
            residual = True,
            dropout=0.1
        )

    def forward(self, x):
        # print(x.size())
        x = x + self.attn(self.norm(x))
        # print(x.size())

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        # print(x.size())
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        # print(x.size())
        x = x.flatten(2).transpose(1, 2)
        # print(x.size())
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)

        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, args= None):
        super(TransMIL, self).__init__()
        
        feature_dim = args.input_dim
        emb_dim = args.emb_dim
        
        self.pos_layer = PPEG(dim=emb_dim)
        self._fc1 = nn.Sequential(nn.Linear(feature_dim, emb_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=emb_dim)
        self.layer2 = TransLayer(dim=emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self._fc2 = nn.Linear(emb_dim, self.n_classes)
        

    def forward(self, x_path, x_omic=None):
        h = x_path
        h = h.float()
        if len(h.shape) == 2:
            h = torch.unsqueeze(h, dim=0)

        h = self._fc1(h)
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h) 
        h = self.norm(h)
        h = self._fc2(h)
        patch_logits = h[:, 1:].squeeze()
        cls_logit = h[:, 0]
        Y_hat = torch.argmax(cls_logit, dim=1)
        Y_prob = F.softmax(cls_logit, dim=1)
        hazards = torch.sigmoid(cls_logit)
        S = torch.cumprod(1 - hazards, dim=1)
        result_dict = {}
        result_dict.update({'hazards': hazards})
        result_dict.update({'S': S})
        # print(hazards.size())
        # print(S.size())
        # assert 2==3

        return hazards, S, Y_hat

