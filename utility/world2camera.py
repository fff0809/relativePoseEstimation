# Copyright (c) 2022, Shining 3D Tech Co., Ltd.
# All rights reserved.
# Function: world coordinate -> camera coordinate.
# Version: 20220413v0.1
# Author: Hanshi Fu

# refer: https://img-blog.csdnimg.cn/20201116145638122.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0Njk3NzM5,size_16,color_FFFFFF,t_70#pic_center
def world2camera(K, coors_world):
    """
    :param RT: [3,4] 相机外参, RT[:,:3]是R, RT[:,3:]是T
    :param coors_world: [N,3] 世界坐标系下的坐标, N是点数
    :return: [N,3] 相机坐标系下的坐标, N是点数
    """
    N = coors_world.shape[0]
    coors_world = np.concatenate((coors_world, np.array([1 for _ in range(N)]).reshape(N,1)), axis=1)# Nx3 -> Nx4
    coors_camera = np.dot(coors_world, RT.transpose())# Nx4 * 4x3 -> Nx3
    return coors_camera

if __name__ == '__main__':
    RT = np.zeros((3, 4))
    coors_world = np.random.randn(10, 3)
    coors_camera = world2camera(K, coors_world)
    print(coors_camera.shape)# (10, 3)
