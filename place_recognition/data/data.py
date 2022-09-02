import os
import os.path as osp
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
from PIL import Image
from interpolate_poses import *

def extract_poses(odom_file, rtime_file):
    fp = open(rtime_file, 'r')
    timestamps = [int(time.strip().split(' ')[0]) for time in fp.readlines()]

    poses = interpolate_vo_poses(odom_file, timestamps, timestamps[0])
    gt_poses = [list(pose[0:3, 3]) for pose in poses]

    with open('./gt_poses.txt', 'w+') as f:
        for i in range(len(gt_poses)):
            f.write(f"{gt_poses[i][0]} {gt_poses[i][1]} {gt_poses[i][2]}\n")

    return gt_poses

def post_process_lidar(points, fwd_range, side_range, angle, gt_pose):
    global counter
    x_points = points[:, 0]
    y_points = points[:, 1]
    i_points = points[:, 3]
    points_copy = copy.deepcopy(points)
    points_copy[:, 3] = np.ones(i_points.shape)

    # plt.scatter(x_points, y_points, s=0.1)
    # plt.show()

    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    m = points_copy[indices]
    rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                        [np.sin(angle), np.cos(angle), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype='object')
    t_mat = np.array([[1, 0, 0, -gt_pose[0]],
                      [0, 1, 0, -gt_pose[1]],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype='object')
    t_mat_inv = np.array([[1, 0, 0, gt_pose[0]],
                          [0, 1, 0, gt_pose[1]],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype='object')
    m = t_mat_inv @ rot_mat @ t_mat @ m.T
    m = m.T

    x_points = m[:, 0]
    y_points = m[:, 1]
    i_points = points[indices, 3]

    res = 0.2
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (x_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (y_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor and ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(fwd_range[0] / res))
    y_img -= int(np.floor(side_range[0] / res))

    # CLIP
    i_points[i_points <= 10] = 0.0
    i_points[i_points > 10] = 255.0
    # pixel_values = np.clip(i_points, a_min=50.0, a_max=150.0)

    def scale_to_255(a, min, max, dtype=np.uint8):
        """ Scales an array of values from specified min, max range to 0-255
            Optionally specify the data type of the output (default is uint8)
        """
        return (((a - min) / float(max - min)) * 255).astype(dtype)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255

    pixel_values = i_points #scale_to_255(pixel_values, min=min(i_points), max=max(i_points))

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


class RadarLidarDataset(Dataset):
    def __init__(self, radar_imdir, odom_file, map_file, rtime_file,
                 radar_transform=None, map_transform=None):
        self.radar_files = sorted([osp.join(radar_imdir, f) for f in os.listdir(radar_imdir)])
        self.num_samples = len(self.radar_files)
        self.odom_file = odom_file
        self.map = np.load(map_file)['arr_0']
        self.radar_transform = radar_transform
        self.map_transform = map_transform
        self.angles = np.random.uniform(0, 2 * np.pi, self.num_samples)
        self.d_xyt = np.array([[-6, -4, -2, 0, 2, 4, 6],
                               [-6, -4, -2, 0, 2, 4, 6],
                               [-6, -4, -2, 0, 2, 4, 6],])
        self.r_xyt = np.array([[-6, 6],
                               [-6, 6],
                               [-6, 6]])

        self.gt_poses = extract_poses(odom_file=self.odom_file, rtime_file=rtime_file)

        self.pos_result_path = './data/pos'
        self.neg_result_path = './data/neg'
        self.radar_path = './data/radar'
        os.makedirs(self.pos_result_path, exist_ok=True)
        os.makedirs(self.neg_result_path, exist_ok=True)
        os.makedirs(self.radar_path, exist_ok=True)

    def __len__(self):
        return len(self.gt_poses)

    def __getitem__(self, idx):
        radar_img = Image.open(self.radar_files[0])

        # Add noise to position
        gt_xyt = np.zeros(3)
        gt_xyt[0] = self.r_xyt[0, 0] + (self.r_xyt[0, 1]-self.r_xyt[0, 0]) * np.random.random()
        gt_xyt[1] = self.r_xyt[1, 0] + (self.r_xyt[1, 1]-self.r_xyt[1, 0]) * np.random.random()
        gt_xyt[2] = self.r_xyt[2, 0] + (self.r_xyt[2, 1]-self.r_xyt[2, 0]) * np.random.random()

        # Location of positive sample
        positive_pose_lidar = copy.deepcopy(self.gt_poses[idx])
        positive_pose_lidar = np.reshape(positive_pose_lidar, 3)
        pose_lidar = np.zeros_like(positive_pose_lidar)

        pose_lidar[2] = positive_pose_lidar[2] #+ float(gt_xyt[2] * np.pi / 180)
        pose_lidar[0] = positive_pose_lidar[0] #+ \
                        # math.cos(positive_pose_lidar[2])*gt_xyt[0] - \
                        # math.sin(positive_pose_lidar[2])*gt_xyt[1]
        pose_lidar[1] = positive_pose_lidar[1] #+ \
                        # math.sin(positive_pose_lidar[2])*gt_xyt[0] + \
                        # math.cos(positive_pose_lidar[2])*gt_xyt[1]

        pose_x, pose_y = pose_lidar[0], pose_lidar[1]

        pos_sample = post_process_lidar(self.map, (pose_x - 60, pose_x + 60),
                                        (pose_y - 60, pose_y + 60), 0.0, [pose_x, pose_y])

        # Extract a negative index
        neg_idx = np.random.randint(0, self.__len__() - 1, 1)[0]
        while True:
            neg_pose = copy.deepcopy(self.gt_poses[neg_idx])
            neg_pose = np.reshape(neg_pose, 3)

            # Check if negative index is far away from positive index
            if np.linalg.norm(neg_pose[0:2] - positive_pose_lidar[0:2]) >= 80.0:
                break
            neg_idx = np.random.randint(0, self.__len__() - 1, 1)[0]

        neg_sample = post_process_lidar(self.map, (neg_pose[0] - 60, neg_pose[0] + 60),
                                        (neg_pose[1] - 60, neg_pose[1] + 60), 0.0, neg_pose[0:2])

        # Apply transforms on all images
        if self.radar_transform is not None:
            radar_img = self.radar_transform(radar_img)

        if self.map_transform is not None:
            pos_sample = self.map_transform(pos_sample)
            neg_sample = self.map_transform(neg_sample)

        return radar_img, pos_sample, neg_sample

if __name__ == '__main__':
    radar_imdir = '../pytorch-CycleGAN-and-pix2pix/oxford/radar'
    odom_file = '../data/oxford/sample/gt/radar_odometry.csv'
    odom_file = './radar_odometry.csv'
    map_file = '../pytorch-CycleGAN-and-pix2pix/map_ox1.npz'
    map_file = '../pytorch-CycleGAN-and-pix2pix/downsampled_map.npz'
    rtime_file = '../data/oxford/sample/radar.timestamps'
    rtime_file = './radar.timestamps'

    dataset = RadarLidarDataset(radar_imdir, odom_file, map_file, rtime_file)

    for idx in range(0, 8000, 5):
        radar_img, pos_sample, neg_sample = dataset.__getitem__(idx)
        # ipdb.set_trace()
        cv2.imwrite(os.path.join(dataset.pos_result_path, f'{idx}.jpg'), pos_sample)
        cv2.imwrite(os.path.join(dataset.neg_result_path, f'{idx}.jpg'), neg_sample)
        cv2.imwrite(os.path.join(dataset.radar_path, f"{idx}.jpg"), np.array(radar_img))
