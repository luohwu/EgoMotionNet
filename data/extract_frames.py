import sys
sys.path.insert(0,'..')
from opt import *
import  os

import pandas as pd
import cv2
import argparse
from ast import literal_eval


def extract_frames_from_video(annotation_path,output_path,video_path):
    clip_ids=args.all_clip_ids
    unavailable_clips=[]
    for i,clip_id in enumerate(clip_ids):

        print(f'{i+1}/{len(clip_ids)} working on clip {clip_id}')

        # mkdir output folder if not existed
        output_folder=os.path.join(output_path,clip_id)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # read annotations and prepare needed frames
        annotations=pd.read_csv(os.path.join(annotation_path,f'{clip_id}.csv'),converters={
            'contact_block':literal_eval,
            'pre_blocks':literal_eval,
        })
        contact_block_list=annotations['contact_block'].tolist()
        contact_frames=[frame for block in contact_block_list for frame in block]
        pred_blocks_list=annotations['pre_blocks'].tolist()
        pred_frames=[frame for blocks in pred_blocks_list for block in blocks for frame in block]


        frames_to_extract=list(set(contact_frames+pred_frames))
        # in-place sort
        frames_to_extract.sort()

        # read videos and extract frames from clip
        video_file=os.path.join(video_path, f'{clip_id}.mp4')
        if not os.path.exists(video_file):
            print(f'clip not found {video_file}')
            unavailable_clips.append(clip_id)
            continue
        # else:
        #     continue

        vidcap=cv2.VideoCapture(video_file)
        assert vidcap.isOpened(),f"failed opening clip {video_file}"
        success=True
        count=0
        while success:
            success,image=vidcap.read()
            count+=1
            assert success, f"failed reading frame: {count} of clip {clip_id}"
            if count in frames_to_extract:
                image=cv2.resize(image,(224,224))
                cv2.imwrite(os.path.join(output_folder,f'frame_{str(count).zfill(10)}.jpg'),image)
                if count==frames_to_extract[-1]:
                    vidcap.release()
                    break

    print(f'unavailable clips:')
    print(unavailable_clips)




if __name__=='__main__':
    extract_frames_from_video(
        annotation_path='/data/luohwu/dataset/EGO4D/ssl_annotations',
        output_path='/data/luohwu/dataset/EGO4D/rgb_frames_ssl',
        video_path='/local/home/luohwu/EGO4D_initial_data/EGO4D_initial_data/v1/clips',
        )






