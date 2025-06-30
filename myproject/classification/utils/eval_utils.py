import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import Maxpool, Meanpool
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_dsmil import *
from models.model_vit import ViT
from models.model_dtfd import Classifier_1fc, DimReduction, Attention_with_Classifier
from models.model_dtfd import Attention_Gated as Attention
from models.model_vit_maxpool import ViTMaxpool
from models.model_transmil import TransMIL
from models.model_abmil import AttentionGated
from models.mhim.model_mhim import MHIM
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from timm.utils import AverageMeter
from collections import OrderedDict
from copy import deepcopy



import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'args': args}
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    elif args.model_type == "transmil":
        model = TransMIL(n_classes=args.n_classes, args=args)
    elif args.model_type == "dsmil":
        i_classifier = FCLayer(out_size=args.n_classes, args=args)
        b_classifier = BClassifier(output_class=args.n_classes, dropout_v=0.0, args=args)
        model = MILNet(i_classifier, b_classifier)
    elif args.model_type == "abmil":
        model = AttentionGated(dropout=args.drop_out)
    elif args.model_type == "maxpool":
        model = Maxpool(n_classes=args.n_classes, args=args)
    elif args.model_type == "meanpool":
        model = Meanpool(n_classes=args.n_classes, args=args)
    elif args.model_type == "vit_maxpool":
        model = ViTMaxpool(n_classes=args.n_classes, args=args)
    elif args.model_type == "vit":
        model = ViT(n_classes=args.n_classes, args=args)
    elif args.model_type == "dtfd":
        classifier = Classifier_1fc(args.mDim, args.n_classes, args.droprate).cuda()
        attention = Attention(args.mDim).cuda()
        dimReduction = DimReduction(1024, args.mDim, numLayer_Res=args.numLayer_Res).cuda()
        model = Attention_with_Classifier(L=args.mDim, num_cls=args.n_classes, droprate=args.droprate_2)

    elif args.model_type in ["mhim", "mhim_pure"]:
        if args.model_type == "mhim_pure":
            if args.baseline == 'attn':
                args.results_dir = os.path.join(args.results_dir, 'ABMIL_tea')
            if args.baseline == 'selfattn':
                args.results_dir = os.path.join(args.results_dir, 'TransMIL_tea')
            model = MHIM(select_mask=False,n_classes=args.n_classes,act=args.act,head=args.n_heads,da_act=args.da_act,baseline=args.baseline).to(device)
        else:
            if args.baseline == 'attn':
                args.results_dir = os.path.join(args.results_dir, 'ABMIL_stu')
            if args.baseline == 'selfattn':
                args.results_dir = os.path.join(args.results_dir, 'TransMIL_stu')
            if args.mrh_sche:
                mrh_sche = cosine_scheduler(args.mask_ratio_h,0.,epochs=args.num_epoch,niter_per_ep=len(train_loader))
            else:
                mrh_sche = None
            model_params = {
                'baseline': args.baseline,
                'dropout': args.dropout,
                'mask_ratio' : args.mask_ratio,
                'n_classes': args.n_classes,
                'temp_t': args.temp_t,
                'act': args.act,
                'head': args.n_heads,
                'msa_fusion': args.msa_fusion,
                'mask_ratio_h': args.mask_ratio_h,
                'mask_ratio_hr': args.mask_ratio_hr,
                'mask_ratio_l': args.mask_ratio_l,
                'mrh_sche': mrh_sche,
                'da_act': args.da_act,
                'attn_layer': args.attn_layer}
            if args.mm_sche:
                mm_sche = cosine_scheduler(args.mm,args.mm_final,epochs=args.max_epochs,niter_per_ep=len(train_loader),start_warmup_value=1.)
            else:
                mm_sche = None
            model = MHIM(**model_params).to(device)
            if args.init_stu_type != 'none':
                if not args.teacher_init.endswith('.pt'):
                    _str = 's_{fold}_checkpoint.pt'.format(fold=cur)
                    _teacher_init = os.path.join(args.teacher_init,_str)
                    print(_teacher_init)
                else:
                     _teacher_init =args.teacher_init
                print('######### Model Initializing.....')
                pre_dict = torch.load(_teacher_init)
                new_state_dict ={}
                if args.init_stu_type == 'fc':
                    for _k,v in pre_dict.items():
                        _k = _k.replace('patch_to_emb.','') if 'patch_to_emb' in _k else _k
                        new_state_dict[_k]=v
                else:
                    info = model.load_state_dict(pre_dict,strict=False)
                    print(info)
                    
            model_tea = deepcopy(model)
            if not args.no_tea_init and args.tea_type != 'same':
                print('######### Teacher Initializing.....')
                try:
                    pre_dict = torch.load(_teacher_init)
                    info = model_tea.load_state_dict(pre_dict,strict=False)
                    print(info)
                except:
                    print('########## Init Error')
            if args.tea_type == 'same':
                model_tea = model
            else:
                model_tea = model
    

    print_network(model)
    
    print(ckpt_path)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.cuda()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    
    correct_idx = []
    
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
            
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error
        
    with open('{}_correct.pkl'.format(args.model_type), 'wb') as f:
        pickle.dump(correct_idx, f)

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
