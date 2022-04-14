# Copyright (c) 2022, Shining 3D Tech Co., Ltd.
# All rights reserved.
# Function: camera coordinate -> word coordinate.
# Version: 20220413v0.1
# Author: hao wang

import numpy as np
from PIL import Image
import os

def camera2word(cam_coordinate,  RT):
    """
    :param cam_coordinate: [num_point, 3]
    :param RT:[4,3] rotation and translation matrix
    """

    #获取相机坐标系下的前三列的每个点的xyz坐标
    cam_coordinate = cam_coordinate[:,0:3]

    #计算相机坐标系下对应的世界坐标
    rt = RT
    ni = np.linalg.inv(rt[0:3, :].T)
    ccld = cam_coordinate - rt[3:, :]
    c2w = np.dot(ccld, ni)
    return c2w

if __name__ == '__main__':
    c_coor = np.array([[-5.99046546, -4.60741416, 130, 168, 121, 105]])
    rt = np.array([[0.999825, 0.0163826, -0.00904598],
                   [-0.0164607, 0.999827, -0.00862799],
                   [0.00890307, 0.00877538, 0.999922],
                   [0.466154, -0.998639, 0.493029]])

    w = camera2word(c_coor, rt)
    print(w)
