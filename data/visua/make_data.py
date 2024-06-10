import numpy as np
from glob import glob
import sys
from tqdm import tqdm

file_list = glob('/home/vimlab/workspace/datasets/NTU_RGB_D/skeleton_npy_cam_ori_track/*')

ntu_skeleton_feature_matrix = None

for k, path in tqdm(enumerate(file_list)):
    skeleton_data = np.load(path,allow_pickle=True).item()
    nbodys = skeleton_data['nbodys'][0]
    if nbodys==1:
        skel_extract_list = ['skel_body0']
        orientation_extract_list = ['orientation_body0']
        tracking_extract_list = ['tracking_body0']
    elif nbodys==2:
        skel_extract_list = ['skel_body0', 'skel_body1']
        orientation_extract_list = ['orientation_body0', 'orientation_body1']
        tracking_extract_list = ['tracking_body0', 'tracking_body1']
    elif nbodys==3:
        skel_extract_list = ['skel_body0', 'skel_body1', 'skel_body2']
        orientation_extract_list = ['orientation_body0', 'orientation_body1', 'orientation_body2']
        tracking_extract_list = ['tracking_body0', 'tracking_body1', 'tracking_body2']
    elif nbodys==4:
        skel_extract_list = ['skel_body0', 'skel_body1', 'skel_body2', 'skel_body3']
        orientation_extract_list = ['orientation_body0', 'orientation_body1', 'orientation_body2', 'orientation_body3']
        tracking_extract_list = ['tracking_body0', 'tracking_body1', 'tracking_body2', 'tracking_body3']


    skel_feature = None
    orientation_feature = None
    tracking_feature = None


    for i, skel in enumerate(skel_extract_list):
        if i==0:
            skel_feature=skeleton_data[skel]
        else:
            skel_feature=np.concatenate([skel_feature, skeleton_data[skel]], axis=0)


    for i, orientation in enumerate(orientation_extract_list):
        if i==0:
            orientation_feature=skeleton_data[orientation]
        else:
            orientation_feature=np.concatenate([orientation_feature, skeleton_data[orientation]], axis=0)


    for i, tracking in enumerate(tracking_extract_list):
        if i==0:
            tracking_feature=skeleton_data[tracking]
        else:
            tracking_feature=np.concatenate([tracking_feature, skeleton_data[tracking]], axis=0)

    one_video_skeletons_feature_matrix = np.concatenate([skel_feature, orientation_feature, tracking_feature], axis=2)
    
    if k==0:
        ntu_skeleton_feature_matrix = one_video_skeletons_feature_matrix
    else:
        ntu_skeleton_feature_matrix = np.concatenate([ntu_skeleton_feature_matrix, one_video_skeletons_feature_matrix], axis=0)

print(ntu_skeleton_feature_matrix.shape)

np.save('/home/vimlab/workspace/datasets/NTU_RGB_D/ntu_skeleton_feature.npy', ntu_skeleton_feature_matrix)