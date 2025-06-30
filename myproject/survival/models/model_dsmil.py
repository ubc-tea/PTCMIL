import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FCLayer(nn.Module):
    def __init__(self, out_size=1, args=None):
        super(FCLayer, self).__init__()
        
        in_size = args.input_dim
                
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats, **kwargs):
        x = self.fc(feats)
        return feats, x

    
class IClassifier(nn.Module):
    def __init__(self, feature_extractor, output_class, args=None):
        super(IClassifier, self).__init__()

        feature_size = args.emb_dim
    
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x, **kwargs):
        device = x.device
        feats = self.feature_extractor(x) 
        c = self.fc(feats.view(feats.shape[0], -1))
        return feats.view(feats.shape[0], -1), c

    
class BClassifier(nn.Module):
    def __init__(self, output_class, dropout_v=0.0, args=None): 
        super(BClassifier, self).__init__()
        
        input_size = args.input_dim
                
        self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )
        
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c, **kwargs): 
        device = feats.device
        V = self.v(feats) 
        Q = self.q(feats).view(feats.shape[0], -1) 
        
        _, m_indices = torch.sort(c, 0, descending=True) 
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        q_max = self.q(m_feats) 
        A = torch.mm(Q, q_max.transpose(0, 1)) 
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) 
        B = torch.mm(A.transpose(0, 1), V)
                
        B = B.view(1, B.shape[0], B.shape[1]) 
        C = self.fcc(B) 
        C = C.view(1, -1)
        return C, A, B 
    
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier, args=None):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.args = args
        
    def forward(self, x_path, x_omic=None, **kwargs):
        # print(x.shape)
        x = x_path
        feats, classes = self.i_classifier(x)
        # print(feats.shape)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        max_prediction, _ = torch.max(classes, 0)
        logits = 0.5 * (prediction_bag + max_prediction)
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        
        result_dict = {}
        # return pred,Y_prob,Y_hat, None, result_dict
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards, S, Y_hat