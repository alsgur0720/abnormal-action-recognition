import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
# import ffmpeg


def joints_dict():
    joints = {
        "coco": {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle"
            },
            "skeleton": [
                # # [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                # # [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # [3, 5], [4, 6]
                [0, 5], [0, 6]
            ]
        },
        "mpii": {
            "keypoints": {
                0: "right_ankle",
                1: "right_knee",
                2: "right_hip",
                3: "left_hip",
                4: "left_knee",
                5: "left_ankle",
                6: "pelvis",
                7: "thorax",
                8: "upper_neck",
                9: "head top",
                10: "right_wrist",
                11: "right_elbow",
                12: "right_shoulder",
                13: "left_shoulder",
                14: "left_elbow",
                15: "left_wrist"
            },
            "skeleton": [
                # [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [13, 3], [12, 2], [13, 12], [13, 14],
                # [12, 11], [14, 15], [11, 10], # [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [3, 6], [2, 6], [6, 7], [7, 8], [8, 9],
                [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10],
            ]
        },
        "kinectV2": {
            "keypoints": {
                0: "spinebase",
                1: "spinemid",
                2: "neck",
                3: "head",
                4: "shoulder_left",
                5: "elbow_left",
                6: "wrist_left",
                7: "hand_left",
                8: "shoulde_rright",
                9: "elbow_right",
                10: "wrist_right",
                11: "hand_right",
                12: "hip_left",
                13: "knee_left",
                14: "ankle_left",
                15: "foot_left",
                16: "hip_right",
                17: "knee_right",
                18: "ankle_right",
                19: "foot_right",
                20: "spine_shoulder",
                21: "headtip_left",
                22: "thumb_left",
                23: "headtip_right",
                24: "thumb_right"
            },
            "skeleton": [
                [0,1],[1,0],[2,20],[3,2],[4,20],
                [5,4],[6,5],[7,6],[8,20],[9,8],
                [10,9],[11,1],[12,0],[13,12],[14,13],
                [15,14],[16,0],[17,16],[18,17],[19,18],
                [20,1],[21,7],[22,7],[23,11],[24,11]
            ]
        },
    }
    return joints


array1 = np.random.randn(13, 25, 8)
array2 = np.random.randn(17, 25, 8)
array3 = np.random.randn(5, 25, 8)

# 세 배열을 병합하여 (22, 25, 8) 차원의 numpy 배열 생성
merged_array = np.concatenate([array1, array2, array3], axis=0)

print(merged_array.shape)