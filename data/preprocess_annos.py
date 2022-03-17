import sys
sys.path.insert(0,'..')
from opt import *
import json
import os
import os.path
import numpy as np
import pandas as pd
import argparse
import cv2

def cal_contact_block(row,num_frames):
  contact_frame=row['contact_frame']
  intervals=[-6,-3,0,3,6]
  contact_block=[min(contact_frame+item,num_frames )for item in intervals]
  return contact_block

def cal_pre_blocks(row,sample_interval=3,num_blocks=5,fps=30):
  contact_frame=row['contact_frame']
  num_frames = num_blocks*0.5*fps
  num_frames=int(num_frames)
  helper=[*range(0,num_frames,sample_interval)]
  helper=[-item for item in helper][::-1]
  end_frame = contact_frame - fps * 0.5
  pre_frames=[int(end_frame+item) for item in helper]
  #ensure idx of frame >0
  pre_frames=[max(1,item) for item in pre_frames]
  frames_per_block=int(len(pre_frames)/num_blocks)
  # pre_frames=1
  pre_blocks=[pre_frames[x:x+frames_per_block] for x in range(0,len(pre_frames),frames_per_block)]

  return pre_blocks





# prepare annotations for self-supervised egocentric representation learning
def make_annotations_for_RL(input_folder,output_folder):
  clip_ids = [f for f in os.listdir(input_folder) if  os.path.isfile(os.path.join(input_folder, f))]
  for i, clip_id in enumerate(clip_ids):
    print(f'{i + 1}/{len(clip_ids)} working on clip: {clip_id}')
    clip_path=os.path.join('/local/home/luohwu/EGO4D_initial_data/EGO4D_initial_data/v1/clips/',f"{clip_id[:-4]}.mp4")
    assert os.path.exists(clip_path),f"{clip_path} not exist"
    vid_cap=cv2.VideoCapture(clip_path)
    num_frames=int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_cap.release()
    original_annotation_file=os.path.join(input_folder,clip_id)
    target_annotation_file=os.path.join(output_folder,clip_id)
    original_annotation=pd.read_csv(original_annotation_file)
    target_annotation=original_annotation.drop_duplicates(subset='contact_frame')[['uid','clip_uid','clip_frame','contact_frame','noun','verb']]
    target_annotation['contact_block']=target_annotation.apply(cal_contact_block,num_frames=num_frames,axis=1)
    target_annotation['pre_blocks']=target_annotation.apply(cal_pre_blocks,axis=1)
    target_annotation.to_csv(target_annotation_file,index=False)
    # target_clip_folder = os.path.join(output_folder, clip_id)
    # if not os.path.exists(target_clip_folder):
    #   os.mkdir(target_clip_folder)
    # images = [image for image in os.listdir(original_clip_folder)]







if __name__=='__main__':
  input_folder=os.path.join(args.data_path,'nao_annotations')
  output_folder=os.path.join(args.data_path,'ssl_annotations')
  make_annotations_for_RL(input_folder,output_folder)

