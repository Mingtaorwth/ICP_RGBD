import numpy as np
import open3d as o3d
import os
from PIL import Image
import glob
import cv2
from collections import Counter, deque
import copy
from track_v8 import process, model
from time import time
import pandas as pd
from ultralytics import YOLO
from multiprocessing import Pool
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import keyboard

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, 
                                                            fx=525, fy=525.,
                                                            cx=319.5, cy=239.5)
    
folder_path_rgb = "apartment/image"
image_files_rgb = glob.glob(os.path.join(folder_path_rgb, '*.jpg'))

folder_path_depth = "apartment/depth"
image_files_depth = glob.glob(os.path.join(folder_path_depth, '*.png'))


voxel_size = 0.02

max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

# option = o3d.pipelines.registration.GlobalOptimizationOption(
#     max_correspondence_distance=max_correspondence_distance_fine,
#     edge_prune_threshold=0.1,
#     reference_node=0)


file_path = 'apartment/apartment.log'  # 替换成实际的文件路径
with open(file_path, 'r') as file:  
    lines = file.readlines()

def refine_registration(s_pcd_down, t_pcd_down, voxel_size, T_ransac):
    distance_threshold = voxel_size * 0.4
    s_pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    t_pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))

    reg_p2p = o3d.pipelines.registration.registration_icp(
        s_pcd_down, t_pcd_down, distance_threshold, T_ransac,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
    return reg_p2p

def preprocess_point_cloud(pcd_down, voxel_size):

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=300))
    return pcd_fpfh

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

def prepare_dataset(voxel_size, s_pcd, t_pcd):
    print(":: Load two point clouds and disturb initial pose.")

    source = s_pcd
    target = t_pcd
    # draw_registration_result(source, target, odo_init, T)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


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

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

vis = o3d.visualization.Visualizer()
vis.create_window()

pose_graph = o3d.pipelines.registration.PoseGraph()


################## Parameters Setting #################
pose_id = 1
step = 1
start = 5800
end = 6800

k = start * 5 + 5
matrix_lines = lines[k + 1:k + 5]
GT_first = np.array([list(map(float, line.split())) for line in matrix_lines])

plt.figure(figsize=(10, 8))
odometry = np.eye(4)
pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
#######################################################

odo_2D_x = []
odo_2D_z = []
gt_x = []
gt_z = []
R_errors = []
t_errors = []

x_label = []

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
    
    # if (pose_id-1) % 31 == 0 and pose_id != 1:
    #     # merged_pcd_down = merged_pcd.voxel_down_sample(voxel_size=voxel_size)

    #     merg_fpfh = preprocess_point_cloud(s_pcd_down, voxel_size)
    #     t_fpfh = preprocess_point_cloud(t_pcd_down, voxel_size)
        
    #     result_ransac = execute_fast_global_registration(s_pcd_down, t_pcd_down,
    #                                     merg_fpfh, t_fpfh,
    #                                     voxel_size)
    #     refine_res = refine_registration(s_pcd_down, t_pcd_down, voxel_size, result_ransac.transformation)
    #     transformation_icp = refine_res.transformation

    # else:
    transformation_icp = pairwise_registration(
            s_pcd_down, t_pcd_down)
    
    
    odometry = np.dot(transformation_icp, odometry)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(
            np.linalg.inv(odometry)))
            
    camera_visualization = o3d.geometry.LineSet.create_camera_visualization(
                    640, 480, pinhole_camera_intrinsic.intrinsic_matrix,
                      np.linalg.inv(pose_graph.nodes[pose_id].pose), scale = 0.02)
    vis.add_geometry(camera_visualization)

    ####################### Plot Ground Truth #########################
    camera_visualization_GT = o3d.geometry.LineSet.create_camera_visualization(
                    640, 480, pinhole_camera_intrinsic.intrinsic_matrix, np.linalg.inv(GT_T), scale=0.02)
    
    num_lines = len(camera_visualization_GT.lines)
    color_gt = np.tile([1.0, 0.0, 0.0], (num_lines, 1))  # [1.0, 0.0, 0.0] 是红色
    camera_visualization_GT.colors = o3d.utility.Vector3dVector(color_gt)
    vis.add_geometry(camera_visualization_GT)
    ###################################################################

    if (pose_id-1) % 30 == 0:
        s_pcd.transform(pose_graph.nodes[pose_id].pose)
        camera_visualization_k = o3d.geometry.LineSet.create_camera_visualization(
                640, 480, pinhole_camera_intrinsic.intrinsic_matrix,
                np.linalg.inv(pose_graph.nodes[pose_id].pose), scale = 0.1)
        
        num_lines = len(camera_visualization_k.lines)
        color_c = np.tile([0.0, 1.0, 0.0], (num_lines, 1))  # [1.0, 0.0, 0.0] 是红色
        camera_visualization_k.colors = o3d.utility.Vector3dVector(color_c)
    
        vis.add_geometry(s_pcd)
        vis.add_geometry(camera_visualization_k)

    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    param.extrinsic = np.linalg.inv(GT_T)
    ctr.convert_from_pinhole_camera_parameters(param)

    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([255, 255, 255])

    odo_2D_x.append(pose_graph.nodes[pose_id].pose[0, -1])
    odo_2D_z.append(pose_graph.nodes[pose_id].pose[2, -1])
    gt_x.append(GT_T[0, -1])
    gt_z.append(GT_T[2, -1])

    R_error, t_error = pose_error(pose_graph.nodes[pose_id].pose[:3, :3],
                                  pose_graph.nodes[pose_id].pose[:3, -1],
                                  GT_T[:3, :3], GT_T[:3, -1]) 

    R_errors.append(R_error)
    t_errors.append(t_error)

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
    pose_id += 1
    k += 5 * step

    if keyboard.is_pressed('q'):  # 按下 'q' 键来停止循环
        print("Stopping the loop...")
        break

vis.run()

    




