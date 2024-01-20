import os
import subprocess
import time
import json
import pdb
import numpy as np

# Specify the folder path for the output npz files
npz_source_folder_path = './GAST-Net-3DPoseEstimation/output'

# Specify the coco anotation JSON file
coco_annotations = '../coco/annotations/person_keypoints_val2017.json'

# Opening JSON file
f = open(coco_annotations)

# load the coco annotations
annotations = json.load(f)

#pdb.set_trace()

# Loop through files in the npz folder
for filename in os.listdir(npz_source_folder_path):
    if filename.endswith('_2D.npz'):  # Check if the file is an mp4 file
        npz_source_file_path = os.path.join(npz_source_folder_path,filename)
        keypoints_detect = np.load(npz_source_file_path)['reconstruction']
        image_filename = f"{filename.split('_2D')[0]}.jpg"
        #pdb.set_trace()
        #image_id = [annotations['images'][cnt]['id'] for cnt in range(len(annotations['images'])) if annotations['images'][cnt]['file_name'] == image_filename]
        image_id=int(filename.split('_2D')[0])
        #data_annotation = [(cnt,image_id,annotations['annotations'][cnt]['id'] ,annotations['annotations'][cnt]['num_keypoints'], image_id,annotations['annotations'][cnt]['keypoints']) for cnt in range(len(annotations['annotations'])) if (annotations['annotations'][cnt]['image_id'] == image_id)]
        data_annotation = [(cnt,image_id,annotations['annotations'][cnt]) for cnt in range(len(annotations['annotations'])) if (annotations['annotations'][cnt]['image_id'] == image_id)]

        if len(data_annotation) > 0:
           print(npz_source_file_path)
           print(len(data_annotation))
           print(data_annotation)
           print(keypoints_detect)
           pdb.set_trace()
        #annotations['annotations'][0]['id']
