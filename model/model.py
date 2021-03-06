import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from backbone.select_backbone import select_resnet
from backbone.convrnn import ConvGRU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class EgoMotionNet(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size, num_seq=6, seq_len=5, pred_step=1, network='resnet18'):
        # sample_size is size of input images, i.e. (128,128)
        super(EgoMotionNet, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 16))
        print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.backbone, self.param = select_resnet(network, track_running_stats=False)
        self.param['num_layers'] = 1 # param for GRU
        self.param['hidden_size'] = self.param['feature_size'] # param for GRU

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])
        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                                )
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)
        self.interaction_bank=nn.Parameter(torch.randn(self.param['feature_size'],self.last_size,self.last_size))
        print(f'interaction bank has size: {self.interaction_bank.shape}')

    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)
        del block
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # before ReLU, (-inf, +inf)
        feature = self.relu(feature) # [0, +inf)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B,N,D,6,6], [0, +inf)
        feature_inf = feature_inf_all[:, N-self.pred_step::, :].contiguous()
        del feature_inf_all

        ### aggregate, predict future ###
        _, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())
        hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step

        pred = []
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:,-1,:]
        pred = torch.stack(pred, 1) # B, pred_step, xxx

        pred_pooled=F.adaptive_avg_pool2d(pred,output_size=(1,1)).squeeze(-1).squeeze(-1)  # B,pred_step, feature_size
        del pred
        pred=torch.einsum('btc,chw->btchw',pred_pooled,self.interaction_bank)

        del hidden


        ### Get similarity score ###
        # from C to D, 3->256 num_channels
        # pred: [B, pred_step, D, last_size, last_size]
        # GT: [B, pred_step, D, last_size, last_size]
        N = self.pred_step
        # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT.
        # pred.shape=(B,N,D,last_size,last_size)
        # pre.permute(0,1,3,4,2)= (B,N,last_size,last_size,D)
        # pre.permute(0,1,3,4,2).contiguous().view(...)=(B*N*last_size^2, D)
        pred = pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*self.last_size**2, self.param['feature_size'])
        feature_inf = feature_inf.permute(0,1,3,4,2).contiguous().view(B*N*self.last_size**2, self.param['feature_size']).transpose(0,1)
        score = torch.matmul(pred, feature_inf).view(B, self.pred_step, self.last_size**2, B, N, self.last_size**2)
        del feature_inf, pred

        if self.mask is None: # only compute mask once
            # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
            mask = torch.zeros((B, self.pred_step, self.last_size**2, B, N, self.last_size**2), dtype=torch.int8, requires_grad=False).detach().to(device)
            mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3 # spatial neg
            for k in range(B):
                mask[k, :, torch.arange(self.last_size**2), k, :, torch.arange(self.last_size**2)] = -1 # temporal neg
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B*self.last_size**2, self.pred_step, B*self.last_size**2, N)
            for j in range(B*self.last_size**2):
                tmp[j, torch.arange(self.pred_step), j, torch.arange(N-self.pred_step, N)] = 1 # pos
            mask = tmp.view(B, self.last_size**2, self.pred_step, B, self.last_size**2, N).permute(0,2,1,3,5,4)
            self.mask = mask

        return [score, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None


class EgoMotionNet_Extractor(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size, num_seq=6, seq_len=5, pred_step=1, network='resnet18'):
        # sample_size is size of input images, i.e. (128,128)
        super(EgoMotionNet_Extractor, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 16))
        print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.backbone, self.param = select_resnet(network, track_running_stats=False)
        self.param['num_layers'] = 1 # param for GRU
        self.param['hidden_size'] = self.param['feature_size'] # param for GRU

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])
        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                                )
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)
        self.interaction_bank=nn.Parameter(torch.randn(self.param['feature_size'],self.last_size,self.last_size))
        print(f'interaction bank has size: {self.interaction_bank.shape}')

    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        feature = self.backbone(block)
        del block
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # before ReLU, (-inf, +inf)
        feature = self.relu(feature) # [0, +inf)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B,N,D,6,6], [0, +inf)
        feature_inf = feature_inf_all[:, N-self.pred_step::, :].contiguous()
        del feature_inf_all

        ### aggregate, predict future ###
        _, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())
        hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step
        context=hidden
        # print(f'shape of context: {context.shape}')
        context=torch.flatten(context,start_dim=1)


        pred = []
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:,-1,:]
        pred = torch.stack(pred, 1) # B, pred_step, xxx
        pred_pooled = F.adaptive_avg_pool2d(pred, output_size=(1, 1)).squeeze(-1).squeeze(
            -1)  # B,pred_step, feature_size
        pred_pooled=torch.softmax(pred_pooled,dim=-1)

        # pred_pooled=F.adaptive_avg_pool2d(pred,output_size=(1,1)).squeeze(-1).squeeze(-1)  # B,pred_step, feature_size
        # pred_pooled=pred_pooled.squeeze(1)


        return [context, pred_pooled.squeeze(1)]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None



class DPC_RNN_Extractor2(nn.Module):
    '''DPC with RNN'''

    def __init__(self, sample_size, num_seq=6, seq_len=5, pred_step=1, network='resnet18'):
        # sample_size is size of input images, i.e. (128,128)
        super(DPC_RNN_Extractor2, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using DPC-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 16))
        print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.backbone, self.param = select_resnet(network, track_running_stats=False)
        self.param['num_layers'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size']  # param for GRU

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'])
        self.network_pred = nn.Sequential(
            nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
        )
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)


    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B * N, C, SL, H, W)
        feature = self.backbone(block)
        del block
        feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=(1, 1, 1))

        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size,
                                       self.last_size)  # before ReLU, (-inf, +inf)
        feature = self.relu(feature)  # [0, +inf)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size,
                               self.last_size)  # [B,N,D,6,6], [0, +inf)
        feature_inf = feature_inf_all[:, N - self.pred_step::, :].contiguous()
        del feature_inf_all

        ### aggregate, predict future ###
        context, hidden = self.agg(feature[:, 0:N - self.pred_step, :].contiguous())
        context = context[:,-1,:].unsqueeze(1) # extract c_t from c_0, ... c_t
        context = F.avg_pool3d(context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1) # context matrix to context vector
        context=context.squeeze(1) #remove time dimention
        # hidden = hidden[:, -1, :]  # after tanh, (-1,1). get the hidden state of last layer, last time step
        #
        # pred = []
        # for i in range(self.pred_step):
        #     # sequentially pred future
        #     p_tmp = self.network_pred(hidden)
        #     pred.append(p_tmp)
        #     _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
        #     hidden = hidden[:, -1, :]
        # pred = torch.stack(pred, 1)  # B, pred_step, xxx
        # del hidden
        #
        # ### Get similarity score ###
        # # from C to D, 3->256 num_channels
        # # pred: [B, pred_step, D, last_size, last_size]
        # # GT: [B, N, D, last_size, last_size]
        # N = self.pred_step
        # # dot product D dimension in pred-GT pair, get a 6d tensor. First 3 dims are from pred, last 3 dims are from GT.
        # # pred.shape=(B,N,D,last_size,last_size)
        # # pre.permute(0,1,3,4,2)= (B,N,last_size,last_size,D)
        # # pre.permute(0,1,3,4,2).contiguous().view(...)=(B*N*last_size^2, D)
        # pred = pred.permute(0, 1, 3, 4, 2).contiguous().view(B * self.pred_step * self.last_size ** 2,
        #                                                      self.param['feature_size'])
        # feature_inf = feature_inf.permute(0, 1, 3, 4, 2).contiguous().view(B * N * self.last_size ** 2,
        #                                                                    self.param['feature_size']).transpose(0, 1)
        # score = torch.matmul(pred, feature_inf).view(B, self.pred_step, self.last_size ** 2, B, N, self.last_size ** 2)
        # del feature_inf, pred
        #
        # if self.mask is None:  # only compute mask once
        #     # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
        #     mask = torch.zeros((B, self.pred_step, self.last_size ** 2, B, N, self.last_size ** 2), dtype=torch.int8,
        #                        requires_grad=False).detach().to(device)
        #     mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3  # spatial neg
        #     for k in range(B):
        #         mask[k, :, torch.arange(self.last_size ** 2), k, :,
        #         torch.arange(self.last_size ** 2)] = -1  # temporal neg
        #     tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B * self.last_size ** 2, self.pred_step,
        #                                                            B * self.last_size ** 2, N)
        #     for j in range(B * self.last_size ** 2):
        #         tmp[j, torch.arange(self.pred_step), j, torch.arange(N - self.pred_step, N)] = 1  # pos
        #     mask = tmp.view(B, self.last_size ** 2, self.pred_step, B, self.last_size ** 2, N).permute(0, 2, 1, 3, 5, 4)
        #     self.mask = mask

        return context,None

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None


if __name__=='__main__':
    device=torch.device('cpu')
    # model = DPC_RNN_Extractor(sample_size=128,
    #                 num_seq=6,
    #                 seq_len=5,
    #                 network='resnet18',
    #                 pred_step=1).to(device)
    # # block: [B, N, C, SL, W, H]
    # input=torch.randn((4,6,3,5,128,128))
    # context,_=model(input)
    # print('context shape:')
    # print(context.shape)

    model = EgoMotionNet(sample_size=128,
                         num_seq=6,
                         seq_len=5,
                         network='resnet18',
                         pred_step=1).to(device)
    # block: [B, N, C, SL, W, H]
    input=torch.randn((4,6,3,5,128,128))
    score,_=model(input)
    print('context shape:')
    print(score.shape)
