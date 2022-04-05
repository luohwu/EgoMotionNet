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

from model.model import DPC_RNN_Extractor,DPC_RNN
from backbone.resnet_2d3d import neq_load_customized
from data.dataset import create_loader
from utils.utils import AverageMeter, save_checkpoint, denorm, calc_topk_accuracy
import time
import pickle
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print(f'using :{torch.cuda.device_count()} GPUs')
    torch.manual_seed(0)
    np.random.seed(0)
    model = DPC_RNN_Extractor(sample_size=128,
                    num_seq=6,
                    seq_len=5,
                    network='resnet18',
                    pred_step=1)
    model=nn.DataParallel(model)
    model=model.to(device)
    global criterion
    criterion = nn.CrossEntropyLoss()


    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    args.old_lr = None
    best_acc = 0
    global iteration
    iteration = 0


    #=================================================
    # pretrain
    args.pretrain=True
    dirname = os.path.dirname(__file__)
    pretrain_model_path=os.path.join(args.exp_path,'EGO4D/SSL/ckpts/model_epoch_200.pth')
    if args.pretrain:
        if os.path.isfile(pretrain_model_path):
            print("=> loading pretrained checkpoint '{}'".format(pretrain_model_path))
            checkpoint = torch.load(pretrain_model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            print("=> loaded pretrained checkpoint '{}' (epoch {})"
                  .format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))

    train_loader = create_loader('train')
    val_loader=create_loader('val')

    #================================================================================================
    #===================================
    #feature data
    #===================================
    # data=extract_features(train_loader,model)
    # print(f'final data length: {len(data)}')
    #
    # pickle_path=os.path.join(args.exp_path,'EGO4D/SSL/data.pkl')
    # with open(pickle_path,'wb') as f:
    #     print('trying to dump data into pickle file')
    #     pickle.dump(data,f)

    pickle_path=os.path.join(args.exp_path,'EGO4D/SSL/data.pkl')
    with open(pickle_path,'rb') as f:
        data=pickle.load(f)


    #================================================================================================
    #===================================
    #similarity matrix
    #===================================
    similarity_matrix_load=True

    if similarity_matrix_load==False:
        similarity_matrix=compute_similarity_matrix(data)
        matrix_path=os.path.join(args.exp_path,'EGO4D/SSL/similarity_matrix.pkl')
        with open(matrix_path,'wb') as f:
            print('dumping similarity matrix')
            pickle.dump(similarity_matrix.cpu().numpy(),f)
    else:
        matrix_path = os.path.join(args.exp_path, 'EGO4D/SSL/similarity_matrix.pkl')
        with open(matrix_path,'rb') as f:
            similarity_matrix=pickle.load(f)

    #================================================================================================
    #===================================
    # start finding nearest neighbours
    #===================================
    for idx in range(11,100):
        nearest_neighbours(data, similarity_matrix, idx=idx)

    print('here')







def extract_features(data_loader, model):
    data_list=[]
    model.eval()
    global iteration

    for idx, data in enumerate(data_loader):
        print(f"current progress: {idx}/{len(data_loader)}, current list length: {len(data_list)}")
        input_seq,input_info=data
        input_seq = input_seq.to(device)
        B = input_seq.size(0)
        features, _ = model(input_seq)
        for i in range(features.shape[0]):
            info=input_info[i]
            feature=features[i].detach().cpu()
            data={}
            data['info']=info
            data['feature']=feature
            data_list.append(data)

    return data_list


def compute_similarity_matrix(data):
    cos=torch.nn.CosineSimilarity(dim=0).to(device)
    num_samples=len(data)
    similarity_matrix=torch.ones(num_samples,num_samples)*100
    similarity_matrix=similarity_matrix.to(device)
    for i in range(num_samples):
        print(f"progress: {i}/{similarity_matrix.shape[0]}")
        query=data[i]['feature'].to(device)
        similarity_matrix[i,i]=-1
        for j in range(num_samples):
            key=data[j]['feature'].to(device)
            if similarity_matrix[j,i]==100:
                similarity_matrix[i,j]=cos(query,key)
            else:
                similarity_matrix[i,j]=similarity_matrix[j,i]
        similarity_matrix[i, i] = -1
    return similarity_matrix

def nearest_neighbours(data,similarity_matrix,idx):
    query=data[idx]['info']
    similarity_vector=similarity_matrix[idx,:]
    top_3_idx=np.argsort(similarity_vector)[-3:][::-1]
    print(f'top 3 idx: {top_3_idx}')

    query_frames=query['contact_block']
    target_foder = os.path.join(args.exp_path, f'EGO4D/SSL/query')
    os.system(f'rm {target_foder}/*.jpg')
    for frame in query_frames:
        image_file=os.path.join(args.frames_path,query['clip_uid'],f'frame_{str(frame).zfill(10)}.jpg')
        target_file=os.path.join(target_foder,f'frame_{str(frame).zfill(10)}.jpg')
        os.system(f'cp {image_file} {target_file}')


    target_foder = os.path.join(args.exp_path, f'EGO4D/SSL/top_3/')
    os.system(f'rm {target_foder}/*.jpg')
    for idx in top_3_idx:
        key=data[idx]['info']
        key_frames=key['contact_block']
        for frame in key_frames:
            image_file = os.path.join(args.frames_path, key['clip_uid'], f'frame_{str(frame).zfill(10)}.jpg')
            target_file = os.path.join(target_foder, f'frame_{str(frame).zfill(10)}.jpg')
            os.system(f'cp {image_file} {target_file}')













main()