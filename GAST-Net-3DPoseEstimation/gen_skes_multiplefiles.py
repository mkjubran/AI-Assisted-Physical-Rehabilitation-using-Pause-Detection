'''
import os
from gen_skes import *

dir_list=os.listdir(r"../../Dataset_CVDLPT_Videos_Blurred_Segments_Corrected")

print(dir_list)

for file in dir_list:
    video = '../../PHVideoDataSet/' + file
    output_3D_npz = './output/' + video.split('/')[-1].split('.')[0] + '_3D.npz'
    if not os.path.exists(output_3D_npz):
       print(video)
       generate_skeletons(video=video, output_animation=False, num_person=1)
       break
'''

import os
import subprocess
import time
import pdb
from gen_skes import *

# Specify the folder path containing long mp4 files
mp4_source_folder_path = '../../Dataset_CVDLPT_Videos_Blurred_Segments_Corrected'

# Specify the folder path containing long mp4 files
npz_destination_folder_path = './output'

# Loop through files in the folder
for filename in os.listdir(mp4_source_folder_path):
    if filename.endswith('.mp4'):  # Check if the file is an mp4 file
        npz_destination_file_path = os.path.join(npz_destination_folder_path,f"{filename.split('.')[0]}_3D.npz")
        mp4_source_file_path = os.path.join(mp4_source_folder_path,filename)
        if not os.path.exists(npz_destination_file_path):
            print(npz_destination_file_path)
            print(filename)
            generate_skeletons(video=mp4_source_file_path, output_animation=False, num_person=1)
