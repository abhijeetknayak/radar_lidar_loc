import os
import open3d as o3d
import cv2
import ipdb
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from torchvision.io import read_image
import matplotlib.pyplot as plt
import copy
import math

from interpolate_poses import *
counter = 0
def extract_poses(odom_file, rtime_file):
    fp = open(rtime_file, 'r')
    timestamps = [int(time.strip().split(' ')[0]) for time in fp.readlines()]

    poses = interpolate_vo_poses(odom_file, timestamps, timestamps[0])
    gt_poses = [list(pose[0:3, 3]) for pose in poses]

    with open('./gt_poses.txt', 'w+') as f:
        for i in range(len(gt_poses)):
            f.write(f"{gt_poses[i][0]} {gt_poses[i][1]} {gt_poses[i][2]}\n")

    return gt_poses

def post_process_lidar(points, fwd_range, side_range):
    global counter
    x_points = points[:, 0]
    y_points = points[:, 1]

    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    if len(list(indices)) < 10:
        ipdb.set_trace()

    x_points = x_points[indices]
    y_points = y_points[indices]

    # plt.scatter(x_points, y_points, s=0.1)
    # plt.savefig(os.path.join('./data_utils/lidar_img', f'{counter}.jpg'))
    # plt.cla()
    # counter += 1

    res = 0.2
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (x_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (y_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # x_img = (x_points // res).astype(np.int32)  # x axis is -y in LIDAR
    # y_img = (y_points // res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor and ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(fwd_range[0] / res))
    y_img -= int(np.ceil(side_range[0] / res))

    # CLIP
    pixel_values = np.ones(x_points.shape) * 255.0#np.clip(i_points, a_min=50.0, a_max=150.0)

    def scale_to_255(a, min, max, dtype=np.uint8):
        """ Scales an array of values from specified min, max range to 0-255
            Optionally specify the data_utils type of the output (default is uint8)
        """
        return (((a - min) / float(max - min)) * 255).astype(dtype)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    # pixel_values = scale_to_255(pixel_values, min=min(i_points), max=max(i_points))

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    y_max = 1 + int((side_range[1] - side_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    x_img = np.clip(x_img, 0, x_max - 1)
    y_img = np.clip(y_img, 0, y_max - 1)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im

def visualize_ply():
    result_path = './lidar_img'
    for file in sorted(os.listdir(result_path))[300:]:
        file_path = os.path.join(result_path, file)
        pcd = o3d.io.read_point_cloud(file_path)
        o3d.visualization.draw_geometries([pcd],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])

class radarLidarDataset(Dataset):
    def __init__(self, radar_imdir, odom_file, map_file, rtime_file,
                 radar_transform=None, map_transform=None):
        self.radar_files = sorted(os.listdir(radar_imdir))
        self.odom_file = odom_file
        self.map = np.load(map_file)['arr_0']
        self.radar_transform = radar_transform
        self.map_transform = map_transform
        self.d_xyt = np.array([[-6, -4, -2, 0, 2, 4, 6],
                               [-6, -4, -2, 0, 2, 4, 6],
                               [-6, -4, -2, 0, 2, 4, 6],])
        self.r_xyt = np.array([[-6, 6],
                               [-6, 6],
                               [-6, 6]])
        pose_file = '../../RaLL/data_utils/gt_poses/RobotCar/pose_xy_01.txt'
        fp = open(pose_file, 'r')
        self.gt_poses = [(float(pose.strip().split(' ')[0]),
                          float(pose.strip().split(' ')[1]),
                          float(pose.strip().split(' ')[2])) for pose in fp.readlines()]

        self.result_path = './data_utils/lidar_img'
        os.makedirs(self.result_path, exist_ok=True)

    def __len__(self):
        return len(self.gt_poses)

    def __getitem__(self, idx):
        # radar_img = self.radar_files[idx]
        # gen lidar-pose randomly in centain range
        # self-motion in robot-frame to world frame
        gt_xyt = np.zeros(3)
        gt_xyt[0] = self.r_xyt[0, 0] + (self.r_xyt[0, 1]-self.r_xyt[0, 0]) * np.random.random()
        gt_xyt[1] = self.r_xyt[1, 0] + (self.r_xyt[1, 1]-self.r_xyt[1, 0]) * np.random.random()
        gt_xyt[2] = self.r_xyt[2, 0] + (self.r_xyt[2, 1]-self.r_xyt[2, 0]) * np.random.random()

        # frame transfer, and note the rad to deg for image rotate
        pose_lidar = copy.deepcopy(self.gt_poses[idx])
        pose_lidar = np.reshape(pose_lidar, 3)

        # align with the affine matrix
        pose_lidar[2] = pose_lidar[2]# + float(gt_xyt[2] * np.pi / 180)
        pose_lidar[0] = pose_lidar[0]# + math.cos(pose_lidar[2])*gt_xyt[0] - math.sin(pose_lidar[2])*gt_xyt[1]
        pose_lidar[1] = pose_lidar[1]# + math.sin(pose_lidar[2])*gt_xyt[0] + math.cos(pose_lidar[2])*gt_xyt[1]

        pose_x, pose_y = pose_lidar[0], pose_lidar[1]
        lidar_img = post_process_lidar(self.map, (pose_x - 40, pose_x + 40),
                                       (pose_y - 40, pose_y + 40))

        return lidar_img

if __name__ == '__main__':
    radar_imdir = '../pytorch-CycleGAN-and-pix2pix/oxford/radar'
    odom_file = '/oxford/sample/gt/radar_odometry.csv'
    map_file = '../pytorch-CycleGAN-and-pix2pix/robotcar_map.npz'
    rtime_file = '/oxford/sample/radar.timestamps'

    dataset = radarLidarDataset(radar_imdir, odom_file, map_file, rtime_file)

    for idx in range(500):
        im = dataset.__getitem__(idx)
        cv2.imwrite(os.path.join(dataset.result_path, f'{idx}.jpg'), im)
