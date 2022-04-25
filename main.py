from comet_ml import Experiment
experiment = Experiment(
    api_key="wU5pp8GwSDAcedNSr68JtvCpk",
    project_name="self-supervised-representation-learning",
    workspace="thesisproject",
)
import os.path

from opt import *
from data.dataset import EGO4D_Dataset
import torch
from torch import nn
from torch import optim
import numpy as np

from model.model import DPC_RNN
from backbone.resnet_2d3d import neq_load_customized
from data.dataset import create_loader
from utils.utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy
import time
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
experiment.log_parameters(args.__dict__)

def main():
    print(f'using :{torch.cuda.device_count()} GPUs')
    torch.manual_seed(0)
    np.random.seed(0)
    model = DPC_RNN(sample_size=128,
                    num_seq=6,
                    seq_len=5,
                    network='resnet18',
                    pred_step=1)
    model=nn.DataParallel(model)
    model=model.to(device)
    global criterion
    criterion = nn.CrossEntropyLoss()


    params = model.parameters()
    optimizer = optim.Adam(params, lr=3e-4, weight_decay=1e-5)
    args.old_lr = None
    best_acc = 0
    global iteration
    iteration = 0


    #=================================================
    # pretrain
    args.pretrain=True
    dirname = os.path.dirname(__file__)
    pretrain_model_path=os.path.join(dirname,'pretrain/k400_128_r18_dpc-rnn.pth.tar')
    if args.pretrain:
        if os.path.isfile(pretrain_model_path):
            print("=> loading pretrained checkpoint '{}'".format(pretrain_model_path))
            checkpoint = torch.load(pretrain_model_path, map_location=torch.device('cpu'))
            model = neq_load_customized(model, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    train_loader = create_loader('train')
    val_loader=create_loader('val')
    print(f'size of train split: {len(train_loader.dataset)}, size of val split: {len(val_loader.dataset)}')
    epoch_save = 20
    for epoch in range( 1000):
        train_loss, train_acc, train_accuracy_list = train(train_loader, model, optimizer, epoch)
        val_loss, val_acc, val_accuracy_list = validate(val_loader, model, epoch)
        print(f"epoch {epoch}:  train loss: {train_loss}, train acc: {train_acc}, val loss: {val_loss}, val acc: {val_acc} ")
        experiment.log_metric( "train top1-acc", train_acc, step=epoch)
        experiment.log_metric( "train loss", train_loss, step=epoch)
        experiment.log_metric( "val top1-acc", val_acc, step=epoch)
        experiment.log_metric( "val loss", val_loss, step=epoch)
        if epoch>0 and epoch%epoch_save==0:
            checkpoint_path = os.path.join('/data/luohwu/experiments/EGO4D/SSL/ckpts', f'model_epoch_{epoch}.pth')
            print(checkpoint_path)

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()
                        },
                       checkpoint_path)


def process_output(mask):
    '''task mask as input, compute the target for contrastive loss'''
    # dot product is computed in parallel gpus, so get less easy neg, bounded by batch size in each gpu'''
    # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
    (B, NP, SQ, B2, NS, _) = mask.size() # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target.requires_grad = False
    return target, (B, B2, NS, NP, SQ)

def train(data_loader, model, optimizer, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.train()
    global iteration

    for idx, data in enumerate(data_loader):
        # print(f"current iteration: {iteration}")
        # print(f"current progress: {idx}/{len(data_loader.dataset)}")
        input_seq,df_item=data
        input_seq = input_seq.to(device)
        B = input_seq.size(0)
        [score_, mask_] = model(input_seq)


        if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

        # score is a 6d tensor: [B, P, SQ, B2, N, SQ]
        # similarity matrix is computed inside each gpu, thus here B == num_gpu * B2
        score_flattened = score_.view(B * NP * SQ, B2 * NS * SQ)
        target_flattened = target_.view(B * NP * SQ, B2 * NS * SQ).to(device)
        target_flattened = target_flattened.to(int).argmax(dim=1)

        loss = criterion(score_flattened, target_flattened)
        top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1, 3, 5))

        accuracy_list[0].update(top1.item(), B)
        accuracy_list[1].update(top3.item(), B)
        accuracy_list[2].update(top5.item(), B)

        losses.update(loss.item(), B)
        accuracy.update(top1.item(), B)
        # experiment.log_metric( "iteration-top1-acc", accuracy.local_avg, step=iteration)
        # experiment.log_metric( "iteration-loss", losses.local_avg, step=iteration)

        del score_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss
        iteration += 1




    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]


def validate(data_loader, model, epoch):
    losses = AverageMeter()
    accuracy = AverageMeter()
    accuracy_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    model.eval()

    with torch.no_grad():
        for idx, data in (enumerate(data_loader)):
            input_seq,df_item=data
            input_seq = input_seq.to(device)
            B = input_seq.size(0)
            [score_, mask_] = model(input_seq)
            del input_seq

            if idx == 0: target_, (_, B2, NS, NP, SQ) = process_output(mask_)

            # [B, P, SQ, B, N, SQ]
            score_flattened = score_.view(B*NP*SQ, B2*NS*SQ)
            target_flattened = target_.view(B*NP*SQ, B2*NS*SQ).to(device)
            target_flattened = target_flattened.to(int).argmax(dim=1)

            loss = criterion(score_flattened, target_flattened)
            top1, top3, top5 = calc_topk_accuracy(score_flattened, target_flattened, (1,3,5))

            losses.update(loss.item(), B)
            accuracy.update(top1.item(), B)

            accuracy_list[0].update(top1.item(),  B)
            accuracy_list[1].update(top3.item(), B)
            accuracy_list[2].update(top5.item(), B)

    return losses.local_avg, accuracy.local_avg, [i.local_avg for i in accuracy_list]



main()