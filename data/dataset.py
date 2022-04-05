import os.path

from comet_ml import Experiment

import sys
sys.path.insert(0,'..')

import time
from ast import literal_eval

import pandas as pd
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from opt import *

import numpy as np
import pickle
from utils.augmentation import *
from torch.utils import data




def make_sequence_dataset(mode='train',dataset_name='EGO4D'):
    print(f'dataset name: {dataset_name}')
    #val is the same as test
    if mode=='all':
        clip_ids=args.all_clip_ids
    elif mode=='train':
        clip_ids = args.train_clip_ids
    else:
        clip_ids=args.val_clip_ids


    print(f'start load {mode} data, #videos: {len(clip_ids)}')
    df_items = pd.DataFrame()
    for video_id in sorted(clip_ids):
        anno_name = video_id + '.csv'
        anno_path = os.path.join(args.annos_path, anno_name)
        if os.path.exists(anno_path):

            img_path = os.path.join(args.frames_path, video_id)


            annos = pd.read_csv(anno_path
                                ,converters={"contact_block": literal_eval,
                                            "pre_blocks": literal_eval,
                                             }
                                )
            annos['img_path']=img_path

            if not annos.empty:
                annos_subset = annos[['img_path', 'contact_block','pre_blocks','clip_uid', 'uid']]
                df_items = df_items.append(annos_subset)


    # df_items=df_items.rename(columns={'class':'category'})
    # df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    print('finished')
    print('=============================================================')
    return df_items


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class EGO4D_Dataset(Dataset):
    def __init__(self, mode='train',dataset_name='EGO4D'):
        self.block_len=5 # how many frams in 1 block
        self.num_blocks=6 # 1 contact block + 5 pre blocks =6 blocks
        self.mode=mode
        self.transform=transforms.Compose([
            RandomSizedCrop(size=128, consistent=True, p=1.0),
            RandomHorizontalFlip(consistent=True),
            RandomGray(consistent=False, p=0.5),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            ToTensor(),
            Normalize()
        ])

        self.data = make_sequence_dataset(mode,dataset_name)


    def __getitem__(self, item):
        # rand_num=torch.rand(1) if self.mode=='train' else 0
        # rand_num=0
        df_item = self.data.iloc[item]
        pre_blocks=df_item['pre_blocks']
        contact_block=df_item['contact_block']
        all_blocks=pre_blocks+[contact_block]

        frame_indices=[frame_idx for block in all_blocks for frame_idx in block]
        pil_images_list=[pil_loader(os.path.join(df_item.img_path,f'frame_{str(idx).zfill(10)}.jpg')) for idx in frame_indices]
        tensor_images_list = self.transform(pil_images_list) # t_seq is a list of tensor (3,128,128)
        (C, H, W) = tensor_images_list[0].size()
        tensor_images = torch.stack(tensor_images_list, 0)
        del tensor_images_list

        tensor_images = tensor_images.view(self.num_blocks, self.block_len, C, H, W).transpose(1, 2)
        return tensor_images,df_item # (num_blocks,C,block_len,H,W) i.e. (6,3,5,128,128)

    def __len__(self):
        return self.data.shape[0]


def my_collate(batch):
    frames_list=[]
    info_list=[]
    for item in batch:
        frames_list.append(item[0])
        info_list.append(item[1])
    return torch.stack(frames_list),info_list


def create_loader(mode='train'):
    dataset=EGO4D_Dataset(mode=mode)
    sampler = data.RandomSampler(dataset)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.bs,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=16,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=my_collate)
    return data_loader

def main_base():
    # train_dataset,val_dataset=ini_datasets(dataset_name='ADL',original_split=False)
    train_dataset = EGO4D_Dataset(mode='test',dataset_name='EGO4D')
    # train_dataset.data.to_csv('/media/luohwu/T7/dataset/EPIC/test.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=4,
                                  num_workers=8, shuffle=True,pin_memory=True,
                                  )
    print(f'start traversing the dataloader')
    start = time.time()
    for epoch in range(1):
        for data in train_dataloader:
            print(1)


    end = time.time()
    print(f'used time: {end-start}')




if __name__ == '__main__':

    main_base()
