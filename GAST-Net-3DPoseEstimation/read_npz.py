import os
from gen_skes import *
import pdb

dir_list=os.listdir(r"../PHVideoDataSet_2D3D/2D/")

print(dir_list)

for file in dir_list:
    filenpz2D = '../PHVideoDataSet_2D3D/2D/' + file
    filenpz3D = '../PHVideoDataSet_2D3D/3D/' + file.split('.')[0][:-2]+'3D.npz'
    npz2D = np.load(filenpz2D)['reconstruction']
    npz3D = np.load(filenpz3D)['reconstruction']
    print(file)
    print(npz2D.shape) 
    print(file.split('.')[0][:-2]+'3D.npz')
    print(npz3D.shape)
    pdb.set_trace()
    #if not os.path.exists(filenpz2D):
