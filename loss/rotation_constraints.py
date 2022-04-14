# Copyright (c) 2022, Shining 3D Tech Co., Ltd.
# All rights reserved.
# Function: calculate rotation_constraints.
# Version: 20220414v0.1
# Author: Hanshi Fu

import numpy as np

def rotation_constraints_single_frame(R_i, E_i, R):
    """
    :param R_i: [3, 3] 对于第i帧, Mask2CAD预测得到
    :param E_i: [3, 4] 相机外参, E_i[:,:3]是R, E_I[:,3:]是T, 外参不是固定的(参考: https://www.jianshu.com/p/2db2b167fb90)
    :param R: [3, 3] 论文方法预测得到
    """
    ER_i = E_i[:, :3]# [3, 3] 相机外参中的R
    rotation_cons = np.linalg.norm(R_i - np.dot(ER_i, R), 2)
    return rotation_cons    

def rotation_constraints_multi_frames(mR_i, mE_i, mR):
    """
    :param mR_i: [m, 3, 3] 对于第i帧, Mask2CAD预测得到, m是帧数
    :param mE_i: [m, 3, 4] 相机外参, E_i[:,:3]是R, E_I[:,3:]是T
    :param mR: [m, 3, 3] 论文方法预测得到
    """
    return sum([rotation_constraints_single_frame(R_i, E_i, R) for R_i, E_i, R in zip(mR_i, mE_i, mR)])


if __name__ == '__main__':
    R_i, Ei, R = np.random.randn(3, 3), np.random.randn(3, 4), np.random.randn(3, 3)
    rotations_cons = rotation_constraints_single_frame(R_i, E_i, R)
    print(rotations_cons)
    
    m = 200
    mR_i, mE_i, mR = np.random.randn(m, 3, 3), np.random.randn(m, 3, 4), np.random.randn(m, 3, 3)
    rotations_cons = rotation_constraints_multi_frames(mR_i, mE_i, mR)
    print(rotations_cons)