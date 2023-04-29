import os
from gen_skes import *

dir_list=os.listdir(r"../../PHVideoDataSet")

print(dir_list)

for file in dir_list:
    video = '../../PHVideoDataSet/' + file
    output_3D_npz = './output/' + video.split('/')[-1].split('.')[0] + '_3D.npz'
    if not os.path.exists(output_3D_npz):
       print(video)
       generate_skeletons(video=video, output_animation=False, num_person=1)
       break
