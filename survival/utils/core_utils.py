from argparse import Namespace
import os

import numpy as np
import torch

from dataset.dataset_generic import save_splits
from utils.utils import get_optim, get_split_loader, get_lr_scheduler


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)



def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    args.writer_dir = os.path.join(args.results_dir, str(cur))
   
    if not os.path.isdir(args.writer_dir):
        os.mkdir(args.writer_dir)
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.writer_dir, flush_secs=15)
    

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    # save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')

    if args.bag_loss == 'ce_surv':
        from utils.utils import CrossEntropySurvLoss
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        from utils.utils import NLLSurvLoss
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        from utils.utils import CoxSurvLoss
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError
    
    
    if args.reg_type == 'omic':
        from utils.utils import l1_reg_all
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':

        from utils.utils import l1_reg_modules
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    args.fusion = None if args.fusion == 'None' else args.fusion

 
  
    if args.model_type == 'ptcmil':
        from models.model_vit_pmt_clu import ViT_PMT_CLU
        model_dict = {'n_classes': args.n_classes,'args': args}
        model = ViT_PMT_CLU(**model_dict)
    elif args.model_type == 'clam':
        from models.model_clam import CLAM_MB
        model_dict = {'n_classes': args.n_classes,'args': args}
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'transmil':
        from models.model_transmil import TransMIL
        model_dict = {'n_classes': args.n_classes,'args': args}
        model = TransMIL(**model_dict)
    elif args.model_type == 'abmil':
        from models.model_abmil import AttentionGated
        model_dict = {'args': args}
        model = AttentionGated(**model_dict)
    elif args.model_type == "ilra":
        from models.model_ilra import ILRA
        model = ILRA(args=args,n_classes=args.n_classes)
    elif args.model_type == "dsmil":
        from models.model_dsmil import FCLayer,BClassifier,MILNet
        i_classifier = FCLayer(out_size=args.n_classes, args=args)
        b_classifier = BClassifier(output_class=args.n_classes, dropout_v=0.0, args=args)
        model = MILNet(i_classifier, b_classifier)

    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.cuda()
    
    if args.load_model:
        model.load_state_dict(torch.load(args.path_load_model))
    print('Done!')
    
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')


    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = False, 
        weighted = False, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = False, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    print('\nInit scheduler...', end=' ')
    lr_scheduler = get_lr_scheduler(args, optimizer, train_loader)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose = True)
    else:
        early_stopping = None
    
    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    latest_c_index = 0.
    max_c_index = 0.
    epoch_max_c_index = 0
    best_val_dict = {}

    print("running with {} {}".format(args.model_type, args.mode))
    for epoch in range(args.start_epoch,args.max_epochs):
       
        from trainer.mil_trainer import train_loop_survival, validate_survival
        train_loop_survival(epoch, model, train_loader, optimizer,lr_scheduler, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc, args)
        val_latest, c_index_val, stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args)
    
        if c_index_val > max_c_index:
            max_c_index = c_index_val
            epoch_max_c_index = epoch
            save_name = 's_{}_checkpoint'.format(cur)
            if args.load_model and os.path.isfile(os.path.join(args.results_dir, save_name+".pt".format(cur))):
                save_name+='_load'
            if os.path.exists(os.path.join(args.results_dir)) == False:
                os.makedirs(os.path.join(args.results_dir))
           
            torch.save(model.state_dict(), os.path.join(args.results_dir, save_name+".pt".format(cur)))
            best_val_dict = val_latest
        

    if args.log_data:
        writer.close()
    print_results = {'result': (max_c_index, epoch_max_c_index)}
    print("================= summary of fold {} ====================".format(cur))
    print("result: {:.4f}".format(max_c_index))
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write('result: {:.4f}, epoch: {}\n'.format(max_c_index, epoch_max_c_index))

    return best_val_dict, print_results