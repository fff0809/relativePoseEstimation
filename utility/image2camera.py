# Copyright (c) 2022, Shining 3D Tech Co., Ltd.
# All rights reserved.
# Function: pixel coordinate -> camera coordinate.
# Version: 20220413v0.1
# Author: hao wang

# from open3d import PointCloud,Vector3dVector,draw_geometries
import numpy as np
from PIL import Image
import os

def image2c(dpt, cam_scale, K):
    """
    :param dpt: [H, W] input depth img
    :param cam_scale: camera scale
    :param K: camera intrinsics
    :return: camera coordinate
    """
    xmap = np.array([[j for i in range(dpt.shape[1])] for j in range(dpt.shape[0])])
    ymap = np.array([[i for i in range(dpt.shape[1])] for j in range(dpt.shape[0])])

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

    # 可视化点云
    # point_cloud = PointCloud()
    # point_cloud.points = Vector3dVector(c_xyz)
    # draw_geometries([point_cloud])
    return c_xyz

if __name__ == '__main__':
    K = np.array([[3538.9, -0.00028937, 323.038],
                  [0, 3541.06, 233.501],
                  [0, 0, 1]])
    with Image.open(os.path.join('depth_path')) as di:
        dpt_mm = np.array(di)
    c = image2c(dpt_mm, cam_scale=1, K=K)
