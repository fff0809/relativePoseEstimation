# Copyright (c) 2022, Shining 3D Tech Co., Ltd.
# All rights reserved.
# Function: camera coordinate -> image coordinate.
# Version: 20220413v0.1
# Author: Hanshi Fu

# refer: https://pic3.zhimg.com/80/v2-98034f3438dc2191e01dfcb8827b86a6_1440w.jpg
def camera2image(K, coors_camera):
    """
    :param K: [3,3] 相机内参
    :param coorsWorld: [N,3] 相机坐标系下的坐标
    :return: [N,2] 像素坐标系下的坐标
    """
    coors_image = np.dot(coors_camera, K.transpose())# Nx3 * 3x3 -> Nx3
    zc = coors_camera[:, -1][:, np.newaxis].repeat(3, -1)# N -> Nx1 -> NX3
    coors_image = coors_image/zc
    return coors_image[:, :2]

if __name__ == '__main__':
    K = np.array([[3538.9, -0.00028937, 323.038],
                  [0, 3541.06, 233.501],
                  [0, 0, 1]])
    coors_camera = np.random.randn(10, 3)
    coors_image = camera2image(K, coors_camera)
    print(coors_image.shape)# (10, 2)