# Copyright (c) 2022, Shining 3D Tech Co., Ltd.
# All rights reserved.
# Function: pixel coordinate -> camera coordinate.
# Version: 20220413v0.1
# Author: hao wang

import numpy as np
from PIL import Image
import os

def rgb2c(dpt, cam_scale, K, rgb):
    """
    :param dpt: [H, W] input depth img
    :param cam_scale: camera scale
    :param K: camera intrinsics
    :param rgb: [H, W, channel] input rgb img
    :return: camera coordinate
    """

    xmap = np.array([[j for i in range(648)] for j in range(488)])
    ymap = np.array([[i for i in range(648)] for j in range(488)])

    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    dpt = dpt.astype(np.float32) / cam_scale
    msk = (dpt > 1e-8).astype(np.float32)
    col = (xmap - K[1][2]) * dpt / K[1][1]
    row = (ymap - K[0][2]) * dpt / K[0][0] - K[0][1] * col
    dpt_3d = np.concatenate(
        (row[..., None], col[..., None], dpt[..., None]), axis=2
    )
    dpt_3d = dpt_3d * msk[:, :, None]
    dpt_3d_reshape = dpt_3d.reshape(dpt_3d.shape[0]*dpt_3d.shape[1],3)
    flatten = dpt_3d_reshape.nonzero()[0].astype(np.uint32)
    c_xyz = dpt_3d_reshape[flatten]

    #把rgb值添加到每个坐标的后三列:x,y,z -> x,y,z,r,g,b
    rgb_reshape = rgb.reshape(rgb.shape[0]*rgb.shape[1],3)
    point_rgb = rgb_reshape[flatten]
    c_cld_rgb = np.c_[c_xyz,point_rgb]

    # 可视化点云
    # point_cloud = PointCloud()
    # point_cloud.points = Vector3dVector(ccld)
    # draw_geometries([point_cloud])
    return c_cld_rgb

if __name__ == '__main__':
    K = np.array([[3538.9, -0.00028937, 323.038],
                  [0, 3541.06, 233.501],
                  [0, 0, 1]])
    with Image.open(os.path.join('depth_path')) as di:
        dpt_mm = np.array(di)
    with Image.open(os.path.join('rgb_path')) as di:
        rgb = np.array(di)
    c = rgb2c(dpt_mm, cam_scale=1, K=K, rgb=rgb)

