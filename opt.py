

import argparse
import os

import pandas as pd


def read_clips(file_path):
    file = open(file_path, 'r')
    file_lines=file.read()
    list=file_lines.split('\n')[:-1]
    return list

def get_noun_categories():
    noun_categories_path=os.path.join(args.data_path,'noun_categories.csv')
    noun_categories=pd.read_csv(noun_categories_path)['name'].tolist()
    return  noun_categories

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--dataset', type=str, default='EGO4D',
                    help='EPIC , ADL, EGO4D')
parser.add_argument('--ait', default=False,action="store_true",
                    help='runing on AIT-server or local computer')

parser.add_argument('--exp_name', default='ssl', type=str,
                    help='experiment path (place to store models and logs)')



parser.add_argument('--debug', default=False,action='store_true', help='debug')

parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.000002, type=float, help='learning rate')

args = parser.parse_args()
if args.ait:
    args.data_path = '/data/luohwu/dataset/EGO4D'
    args.annos_path = '/data/luohwu/dataset/EGO4D/ssl_annotations'
    args.frames_path = '/data/luohwu/dataset/EGO4D/rgb_frames_ssl'
    args.exp_path = '/data/luohwu/experiments'
else:
    args.data_path = '/home/luohwu/ait-data/dataset/EGO4D'
    args.annos_path = '/home/luohwu/ait-data/dataset/EGO4D/ssl_annotations'
    args.frames_path = '/home/luohwu/ait-data/dataset/EGO4D/rgb_frames_ssl'
    args.exp_path = '/home/luohwu/ait-data/experiments'


# args.data_path='/media/luohwu/T7/dataset/EGO4D/' if args.euler==False else os.path.join(os.environ['TMPDIR'],'dataset','EGO4D')
# # args.data_path='/home/luohwu/nobackup/training/dataset/EGO4D' if args.euler==False else '/cluster/work/hilliges/luohwu/nobackup/training/dataset/EGO4D'
# args.exp_path='/home/luohwu/euler/experiments' if args.euler==False else '/cluster/home/luohwu/experiments'
#
# # args.data_path='/media/luohwu/T7/dataset/EGO4D/' if args.euler==False else '/cluster/work/hilliges/luohwu/nobackup/training/dataset/EGO4D'


if args.debug:
    # train_clip_id = {'P_01', 'P_02', 'P_03', 'P_04', 'P_05', 'P_06'}
    args.train_clip_ids = ['ffc0c4fd-372d-42dd-83a2-5c456551ed13']
    args.val_clip_ids = ['ffc0c4fd-372d-42dd-83a2-5c456551ed13']
else:
    args.train_clip_ids = read_clips(os.path.join(args.data_path,'train_clips.txt'))
    args.val_clip_ids = read_clips(os.path.join(args.data_path,'val_clips.txt'))

args.all_clip_ids = args.train_clip_ids + args.val_clip_ids
args.noun_categories=get_noun_categories()

# annos_path = 'nao_annotations'
# frames_path = 'rgb_frames_resized'  #
# args.annos_path=os.path.join(args.data_path,annos_path)
# if args.euler:
#     args.annos_path=os.path.join('/cluster/work/hilliges/luohwu/nobackup/training/dataset/EGO4D',annos_path)
# else:
#     args.annos_path=os.path.join('/home/luohwu/nobackup/training/dataset/EGO4D',annos_path)
# args.frames_path=os.path.join(args.data_path,frames_path)







if __name__=='__main__':
    print(f'original split? {args.original_split}')
    if args.euler:
        print(f'using euler')
    else:
        print(f'using local ')