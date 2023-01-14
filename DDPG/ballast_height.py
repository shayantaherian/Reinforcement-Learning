# Imports
import os
import json
import configparser
import csv
import logging

import numpy as np

from libs import log, reproject
# from libs.debug.debug import DEBUG
# from detect_violations import DetectedViolations

from refactor_libs.geometry_utils import GeometryUtils
from refactor_libs import violation_rulesets
from refactor_libs import label_info
from refactor_libs.point_cloud_utils import PointCloudUtils

# from scipy.signal import find_peaks
import glob
import cv2
import open3d as o3d
from os.path import exists
from skimage.measure import LineModelND, ransac

SEMSEG_PATH = "./data/semseg/"
RECONSTRUCTION_PATH = "./data/undistorted"

POINTCLOUD_WIDTH = 400
POINTCLOUD_HEIGHT = 255

LABEL_KEY = {
    "background": 0,
    "rail": 10,
    "sign": 20,
    "signal": 19,
    "vegetation": [6, 21, 22],
    "grass": 6,
    "bushes": 21,
    "trees": 22,
    "poles": 17,
}


def get_seg_im_path(idx):
    # 'data_2/semseg/rgb/000321.jpeg.jpg.png'
    return f"{SEMSEG_PATH}/rgb/{idx:06d}.jpeg.jpg.png"


def get_ply_path(idx):
    # 'data_2/semseg/rgb/000321.jpeg.jpg.png'
    return f"{RECONSTRUCTION_PATH}/depthmaps/{idx:06d}.jpeg.clean.ply"
  

def get_frame_number(path):
    return path.split('/')[-1].split('.')[0]


def get_pc_size(path):
    im = cv2.imread(path)
    return im.shape[0], im.shape[1]

def main():
    point_cloud_dir = f"{RECONSTRUCTION_PATH}/depthmaps/*.ply"
    ply_paths = sorted(glob.glob(point_cloud_dir))

    reconstruction_json_dir = f"{RECONSTRUCTION_PATH}/reconstruction.json"
    with open(reconstruction_json_dir, "r") as file:
        reconstruction_param = json.load(file)

    image_dir = f"{RECONSTRUCTION_PATH}/images/*.jpg"
    image_paths = sorted(glob.glob(image_dir))
    camera = reproject.load_camera(reconstruction_param)
    shots = reproject.parse_shots(reconstruction_param)

    # semantic segmentation labelling
    semseg_dir = f"{SEMSEG_PATH}/int/*.png"
    semseg_paths = sorted(glob.glob(semseg_dir))

    # TODO: do this check in the main thing
    image_height, image_width = get_pc_size(image_paths[0])

    print(f"PC height: {image_height}, width: {image_width}")

    current_cloud_chunk = []

    # Need to find size of an image.


    for i, ply_path in enumerate(ply_paths):
        this_cloud = o3d.io.read_point_cloud(ply_path, format='xyzrgb')
        this_points = np.asarray(this_cloud.points)
        this_cloud.colors = o3d.utility.Vector3dVector(np.asarray(this_cloud.colors) / 255)

        # o3d.visualization.draw_geometries([this_cloud], window_name='This Cloud')

        current_cloud_chunk.append(this_cloud)

        this_points = reproject.load_pointcloud(ply_paths[i])
        coords_2d = camera.reproject_coordinates(shots[i], this_points, out_of_bounds_error=False)


        frame_id = int(ply_path.split('/')[-1].split('.')[0])

        # Get label colors
        im_path = get_seg_im_path(frame_id)
        im = cv2.imread(im_path)
        im = cv2.resize(im, (image_width, image_height))

        this_points_points = this_points.coordinates
        this_points_points = this_points_points[:, coords_2d[0, :] < image_width]
        coords_2d = coords_2d[:, coords_2d[0, :] < image_width]
        this_points_points = this_points_points[:, coords_2d[1, :] < image_height]
        coords_2d = coords_2d[:, coords_2d[1, :] < image_height]

        labels_colors = im[coords_2d[1], coords_2d[0], :]/255
        label_pc = o3d.geometry.PointCloud()
        label_pc.colors = o3d.utility.Vector3dVector(labels_colors)
        label_pc.points = o3d.utility.Vector3dVector(this_points_points.transpose())

        o3d.visualization.draw_geometries([label_pc], window_name='This Cloud')


if __name__ == '__main__':

    # Launch main function, possibly in a class?
    main()
