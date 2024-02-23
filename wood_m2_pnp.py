import numpy as np
import open3d as o3d
import os
from PIL import Image
import glob
import cv2
from collections import Counter, deque
import copy
from time import time
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from feature_funcs import *

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, 
                                                            fx=525, fy=525.,
                                                            cx=319.5, cy=239.5)

def pose_error(estimated_R, estimated_t, ground_truth_R, ground_truth_t):
    """
    计算位姿误差

    参数：
    estimated_R: 估计的旋转矩阵
    estimated_t: 估计的平移向量
    ground_truth_R: 地面真值的旋转矩阵
    ground_truth_t: 地面真值的平移向量

    返回：
    rotation_error: 旋转矩阵之间的角度差
    translation_error: 平移向量之间的欧氏距离
    """
    # 计算旋转矩阵之间的角度差
    rotation_error = np.arccos((np.trace(np.dot(estimated_R.T, ground_truth_R)) - 1) / 2)
    
    # 计算平移向量之间的欧氏距离
    translation_error = np.linalg.norm(estimated_t - ground_truth_t)

    return rotation_error, translation_error

def numeric_sort(filename):
    return int(os.path.basename(filename).split('.')[0])

folder_path_rgb = "apartment/image"
image_files_rgb = glob.glob(os.path.join(folder_path_rgb, '*.jpg'))
image_files_rgb = sorted(image_files_rgb, key=numeric_sort)

folder_path_depth = "apartment/depth"
image_files_depth = glob.glob(os.path.join(folder_path_depth, '*.png'))
image_files_depth = sorted(image_files_depth, key=numeric_sort)

def vector6(x, y, z, a, b, c):
    """Create 6d double numpy array."""
    return np.array([x, y, z, a, b, c], dtype=np.float32)

voxel_size = 0.02

max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

file_path = 'apartment/apartment.log'  # 替换成实际的文件路径
with open(file_path, 'r') as file:  
    lines = file.readlines()


def pairwise_registration(source, target):
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=80))
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=80))

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    transformation_icp = icp_fine.transformation
    return transformation_icp

vis = o3d.visualization.Visualizer()
vis.create_window()

################## Parameters Setting #################
pose_id = 1
step = 1
start = 5800
end = 6800

k = start * 5 + 5
matrix_lines = lines[k + 1:k + 5]
GT_first = np.array([list(map(float, line.split())) for line in matrix_lines])

odometry = np.eye(4)
#######################################################

odo_2D_x = []
odo_2D_z = []
gt_x = []
gt_z = []
R_errors = []
t_errors = []

x_label = []
transformation_pnp = np.eye(4)
plt.figure(figsize=(10, 8))
for source_id in tqdm(range(start, end, step)):
    GT_lines = lines[k + 1:k + 5] # Groud Truth
    GT_T = np.linalg.inv(GT_first) @ np.array([list(map(float, line.split())) for line in GT_lines])
    # Groud Truth

    if source_id == start:

        s_color_raw = o3d.io.read_image(image_files_rgb[source_id])
        s_depth_raw = o3d.io.read_image(image_files_depth[source_id])
        s_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(s_color_raw, s_depth_raw, 
                                                                                convert_rgb_to_intensity=False)
        s_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        s_rgbd_image,
                        o3d.camera.PinholeCameraIntrinsic(
                            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        
    s_pcd_down = s_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    target_id = source_id + step
        
    t_color_raw = o3d.io.read_image(image_files_rgb[target_id])
    t_depth_raw = o3d.io.read_image(image_files_depth[target_id])
    t_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(t_color_raw, t_depth_raw, 
                                                                        convert_rgb_to_intensity=False)

    t_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                t_rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    t_pcd_down = t_pcd.voxel_down_sample(voxel_size=voxel_size)
    # transformation_icp = pairwise_registration(
    #     s_pcd_down, t_pcd_down)
    
    img_1 = np.array(s_rgbd_image.color)
    img_2 = np.array(t_rgbd_image.color)

    feature_s, feature_t = find_features(img_1, img_2)

    pcd_f_s = map_2d_to_3d(s_rgbd_image, pinhole_camera_intrinsic.intrinsic_matrix, feature_s)

    rot_matrix, tran_vector, cm_points_2, points_3d = PnP(pcd_f_s, feature_t, pinhole_camera_intrinsic.intrinsic_matrix,
                                                                         np.zeros((5, 1), dtype=np.float32))
    
    P_pnp = np.concatenate((rot_matrix, tran_vector), axis=1)
    transformation_pnp[:3, :4] = P_pnp
    

    # point_cloud_f_s = o3d.geometry.PointCloud()
    # point_cloud_f_s.points = o3d.utility.Vector3dVector(pcd_f_s)

    # point_cloud_f_t = o3d.geometry.PointCloud()
    # point_cloud_f_t.points = o3d.utility.Vector3dVector(pcd_f_t)
    # o3d.visualization.draw_geometries([point_cloud_f_s, point_cloud_f_t])
    

    odometry = np.dot(transformation_pnp, odometry)

    camera_visualization = o3d.geometry.LineSet.create_camera_visualization(
                    640, 480, pinhole_camera_intrinsic.intrinsic_matrix,
                      odometry, scale = 0.02)
    vis.add_geometry(camera_visualization)

    ####################### Plot Ground Truth #########################
    camera_visualization_GT = o3d.geometry.LineSet.create_camera_visualization(
                    640, 480, pinhole_camera_intrinsic.intrinsic_matrix, np.linalg.inv(GT_T), scale=0.02)
    
    num_lines = len(camera_visualization_GT.lines)
    color_gt = np.tile([1.0, 0.0, 0.0], (num_lines, 1))  # [1.0, 0.0, 0.0] 是红色
    camera_visualization_GT.colors = o3d.utility.Vector3dVector(color_gt)
    vis.add_geometry(camera_visualization_GT)
    ###################################################################

    if pose_id % 30 == 0:
        s_pcd_down.transform(np.linalg.inv(odometry))
        camera_visualization_k = o3d.geometry.LineSet.create_camera_visualization(
                640, 480, pinhole_camera_intrinsic.intrinsic_matrix,
                odometry, scale = 0.1)
        
        num_lines = len(camera_visualization_k.lines)
        color_c = np.tile([0.0, 1.0, 0.0], (num_lines, 1))  # [1.0, 0.0, 0.0] 是红色
        camera_visualization_k.colors = o3d.utility.Vector3dVector(color_c)
    
        vis.add_geometry(s_pcd_down)
        vis.add_geometry(camera_visualization_k)

    ctr = vis.get_view_control()
    ctr.rotate(0, 1000)
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([255, 255, 255])

    R_error, t_error = pose_error(np.linalg.inv(odometry)[:3, :3], np.linalg.inv(odometry)[:3, -1], GT_T[:3, :3], GT_T[:3, -1]) 

    R_errors.append(R_error)
    t_errors.append(t_error)

    odo_2D_x.append(np.linalg.inv(odometry)[0, -1])
    odo_2D_z.append(np.linalg.inv(odometry)[2, -1])
    gt_x.append(GT_T[0, -1])
    gt_z.append(GT_T[2, -1])
    x_label.append(pose_id-1)

    plt.subplot(2, 2, 1)
    plt.cla() 
    plt.plot(odo_2D_x, odo_2D_z, c='b')
    plt.plot(gt_x, gt_z, c='r')
    plt.legend(["Odometry", "GT"])
    plt.title("Trajectory 2D/ m")

    plt.subplot(2, 2, 2)
    plt.cla() 
    plt.plot(x_label, R_errors, c='g')
    plt.title("Rotation error/ radians")

    plt.subplot(2, 2, 3)
    plt.cla() 
    plt.plot(x_label, t_errors, c='orange')
    plt.title("Translation error/ m")

    plt.pause(0.001)

    vis.poll_events()
    vis.update_renderer()
    s_pcd = t_pcd
    s_rgbd_image = t_rgbd_image
    pose_id += 1
    k += 5 * step

vis.run()

    




