import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .nystrom_attention_custom import NystromAttention_custom
from nystrom_attention import NystromAttention
import random
from .tcformer.tcformer_layers import Block, TCBlock, OverlapPatchEmbed, CTM, CTM_w_clustering
from .tcformer.tcformer_utils import (
    merge_tokens, cluster_dpc_knn, token2map,
    map2token, token_downup, sra_flops, map2token_flops, token2map_flops, downup_flops, cluster_and_merge_flops)
import torch.fft as fft
from .dec import DEC

def soft_merge(assn_matrix, x):
    merge_token = assn_matrix.squeeze(0).transpose(1,0) @ x
    merge_token = merge_token.unsqueeze(0)
    return merge_token
        
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, heads=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//heads,
            heads = heads,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        attn = self.attn(self.norm(x))
        x = x + attn
        return x

class TransLayer_custom(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, heads=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention_custom(
            dim = dim,
            dim_head = dim//heads,
            heads = heads,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )
        


        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         PreNorm(dim, NystromAttention_custom(
        #     dim = dim,
        #     dim_head = dim//heads,
        #     heads = heads,
        #     num_landmarks = dim//2,    # number of landmarks
        #     pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        #     residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
        #     dropout=0.1)),
        #         PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
        #     ]))


    def forward(self, x):
        attn,pmt_tokens,assn_matrix,idx_cluster = self.attn(self.norm(x))
        x = x + attn
        return x, pmt_tokens,assn_matrix,idx_cluster

class ViT_PMT_CLU(nn.Module):
    def __init__(self, n_classes, top_k=1, n_layers=2, args=None):
        super(ViT_PMT_CLU, self).__init__()
        
        input_dim = args.input_dim
        emb_dim = args.emb_dim
        self.cluster_number = args.cluster_number
        
        self._fc1 = nn.Sequential(nn.Linear(input_dim, emb_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, emb_dim))

        # self.pmt_token = nn.Parameter(torch.randn(1, args.cluster_number, emb_dim))
        
        initial_cluster_centers = torch.zeros(args.cluster_number, emb_dim, dtype=torch.float)
        nn.init.xavier_uniform_(initial_cluster_centers)
        orthogonal_cluster_centers = torch.zeros(args.cluster_number, emb_dim, dtype=torch.float)
        orthogonal_cluster_centers[0] = initial_cluster_centers[0]/torch.norm(initial_cluster_centers[0], p=2)
        for i in range(1, args.cluster_number):
            project = 0
            for j in range(i):
                project += self.project(
                    initial_cluster_centers[j], initial_cluster_centers[i])
            initial_cluster_centers[i] -= project
            orthogonal_cluster_centers[i] = initial_cluster_centers[i] / \
                torch.norm(initial_cluster_centers[i], p=2)
        self.pmt_token  = orthogonal_cluster_centers.unsqueeze(0)
        self.pmt_token = nn.Parameter(self.pmt_token)

        self.n_classes = n_classes
        trans_layers0 = [TransLayer(dim=emb_dim, heads=args.cluster_number) for _ in range(args.n_layers-1)]
        self.layer0 = nn.Sequential(*trans_layers0)
        
        trans_layers1 = [TransLayer_custom(dim=emb_dim,heads=args.cluster_number)]
        self.layer0 = nn.Sequential(*trans_layers0)
        self.layer1 = nn.Sequential(*trans_layers1)
        
        self.norm = nn.LayerNorm(emb_dim)
        self._fc2 = nn.Linear(emb_dim, self.n_classes)
        self.dim_reduction = nn.Conv1d(args.cluster_number+1, 1, kernel_size=1)
        self._fc3 = nn.Linear(emb_dim, self.n_classes)
        # self._fc3 = nn.Linear(emb_dim*self.cluster_number, self.n_classes)
        self.fc_out = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, self.n_classes)
        )
        self._fc4 = nn.Linear(emb_dim, self.n_classes)
        self.identity = torch.eye(self.pmt_token.size(1), self.pmt_token.size(1)).cuda()
        
        self.args = args

        self.ctm = CTM_w_clustering(args.cluster_number, emb_dim)
        encoder_hidden_size = 32
        self.encoder = nn.Sequential(
                nn.Linear(emb_dim , encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,emb_dim),
            )
        # self.dec = DEC(cluster_number=args.cluster_number, hidden_dimension=emb_dim,encoder=self.encoder)
    @staticmethod
    def project(u, v):
        return (torch.dot(u, v)/torch.dot(u, u))*u
        
    def forward(self, x_path,x_omic=None, result_dict=None):
        h = self._fc1(x_path) 

        #---->prompt_token
        if result_dict is None: # the first iteration
            pmt_tokens = self.pmt_token.cuda() # random initialization
        else:
            pmt_tokens = result_dict['pmt_tokens']
        
        # h = torch.cat((torch.reshape(pmt_tokens, (1,h.size(-1) )), h), dim=0)
        # print(pmt_tokens.size())
        pmt_tokens = pmt_tokens.squeeze(0)
        h = torch.cat((pmt_tokens, h), dim=0)
        #---->cls_token
        cls_tokens = self.cls_token.cuda()
        h = torch.cat((cls_tokens, h), dim=0)

        #---->global Translayer
        h = torch.unsqueeze(h, dim=0)
        # h = self.layer0(h)
        
        #---->Translayer prompt clustering
        h,new_pmt_tokens,assn_matrix,idx_cluster = self.layer1(h)

        # print(new_pmt_tokens.size())
        #---->moving_ave_prompt_token
        if self.args.moving_pmt:
            if result_dict is None:
                new_pmt_tokens = new_pmt_tokens
            else:
                new_pmt_tokens = (1-self.args.momentum) * pmt_tokens + self.args.momentum * new_pmt_tokens
                # print(new_pmt_tokens.requires_grad)
        else:
            new_pmt_tokens = new_pmt_tokens
        #---> local Translayer
        h_local = []
        p_global = []
        idx_cluster_new = []
        cls_token = h[:,0,:].unsqueeze(0)
        ''' # use cls in each local transfromer
        for i in range(self.cluster_number):
            if sum(idx_cluster==i) == 0:
                continue
            p = new_pmt_tokens[:,i,:].unsqueeze(0)
            h_ = h[:,self.cluster_number+1:,:][:,idx_cluster==i,:]
            h_ = torch.cat((cls_token,p,h_),dim=1) # [1, cls||p_i||N_i, emb_dim]
            h_ = self.layer0(h_)[:,2:,:] # [1, N_i, emb_dim]
            h_local.append(h_)
            idx_cluster_new.append([i]*(h_.size(1)))
            cls_token = h_[:,0,:].unsqueeze(1)
        h_global = torch.cat( h_local, dim=1) # [1, N, emb_dim]
        '''
        # not use cls in each local transfromer
        for i in range(self.cluster_number):
            p = new_pmt_tokens[:,i,:].unsqueeze(0)
            h_ = h[:,self.cluster_number+1:,:][:,idx_cluster==i,:]
            h_ = torch.cat((p,h_),dim=1)
            h_local.append(self.layer0(h_)[:,1:,:])
            idx_cluster_new.append([i]*(h_.size(1)-1))
            p_global.append(p)
        h_global = torch.cat(h_local, dim=1)
        
        idx_cluster_new = sum(idx_cluster_new, [])
        idx_cluster_new = torch.tensor(idx_cluster_new).cuda()

        h = torch.cat((cls_token,h_global),dim=1) # [1, 1+N, emb_dim]
        h = torch.squeeze(h, dim=0) # [1+N, emb_dim]
        #---->cls_token
        cls_token_output = self.norm(h)[0]
        y_outputs = self.norm(h)[1:]

        h_before_merge = h[1:,:] # [N, emb_dim]
        #---->merge token to involve prediction
        if self.args.merge_token:
            # assert 2==3
            #----> hard merge
            # print(idx_cluster_new.size())
            merge_token = self.ctm(h_before_merge,idx_cluster_new)
            #----> soft merge, hyper cluster
            # merge_token = soft_merge(assn_matrix,h_before_merge)
            #----> soft merge, from bnt
            # merge_token, assignment = self.dec(h, new_pmt_tokens)
            # merge_token = merge_token.reshape(1,1,-1)



        #---->orthogonal of prompt_token
        # print(pmt_tokens.size())
        
        PTC_loss = torch.norm(new_pmt_tokens @ torch.transpose(new_pmt_tokens, 1, 2) - self.identity)
       
        
        #---->predict
        if self.args.merge_token:
            ##---->mean cls and merge token to involve prediction
            cls_token_output = cls_token_output.unsqueeze(0)
            token_output = torch.cat((cls_token_output, merge_token.squeeze(0)), dim=0)

                # ----> Ablation: prompt as prototypes
            # token_output = torch.cat((cls_token_output, new_pmt_tokens.squeeze(0)), dim=0)
                # ----> Ablation: w/o cls token
            # token_output = merge_token.squeeze(0)
                # ----> Ablation: cls token only
            # token_output = cls_token_output

            mean_token_output = torch.mean(token_output, dim=0)
            # mean_token_output = self.dim_reduction(token_output).unsqueeze(0)
            # self.mean_token_output = mean_token_output
            # print(mean_token_output.size())
            logit = self._fc3(mean_token_output).squeeze(0)
            # logit = self.fc_out(mean_token_output)
            #---->mean merge token to predict, average logits with cls token
            # merge_token = torch.mean(merge_token.squeeze(), dim=0)
            # # print(merge_token.size())
            # cls_logit = self._fc2(cls_token_output)
            # merge_logit = self._fc3(merge_token).squeeze()
            # logit = torch.divide(cls_logit + merge_logit,2)            
            #---->max merge token to predict, average logits with cls token
            # cls_logit = self._fc2(cls_token_output)
            # merge_logit = self._fc3(merge_token.squeeze(0))
            # merge_logit, _ = torch.max(merge_logit, dim=0)
            # logit = torch.divide(cls_logit + merge_logit,2)            
            # #---->max merge token to predict
            # merge_logit = self._fc3(merge_token.squeeze(0))
            # logit, _ = torch.max(merge_logit, dim=0)
            ##---->prototype token to predict
            # logit = self._fc4(pmt_tokens).squeeze(0)
            # print(logit)
            # logit, _ = torch.mean(logit, dim=0)

            Y_hat = torch.argmax(logit, dim=0).reshape(1, 1)
            Y_prob = F.softmax(logit, dim=0).unsqueeze(dim=0)
            logit = logit.unsqueeze(0)
        else:
            #----> cls token to predict
            # logit = self._fc2(cls_token_output) #[n_classes]
            # Y_hat = torch.argmax(logit, dim=0).reshape(1, 1)
            # Y_prob = F.softmax(logit, dim=0).unsqueeze(dim=0)
            #----> max output token to predict
            logit = self._fc3(y_outputs)
            y_probs = F.softmax(logit, dim=1)
            top_instance_idx = torch.topk(y_probs[:, 1], 1, dim=0)[1].view(1,)
            top_instance = torch.index_select(logit, dim=0, index=top_instance_idx)
            logit = top_instance
            Y_hat = torch.topk(top_instance, 1, dim=1)[1]
            Y_prob = F.softmax(top_instance, dim=1) 

        result_dict = {}
        
        # result_dict.update({'all_instance': patch_logits})
        new_pmt_tokens = new_pmt_tokens.detach()
        result_dict.update({'PTC_loss': PTC_loss})
        result_dict.update({'pmt_tokens': new_pmt_tokens})
        result_dict.update({'h_before_merge': h_before_merge})
        result_dict.update({'idx_cluster': idx_cluster})
        
        # return logit, Y_prob, Y_hat, None, result_dict
        hazards = torch.sigmoid(logit)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, Y_hat,result_dict

