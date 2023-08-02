import open3d as o3d
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotate_point_cloud(point_cloud, center, angles_deg):
    # Translate the point cloud to center the object at the origin
    translated_points = point_cloud.points - center

    # Convert rotation angles from degrees to radians
    angles_rad = np.radians(angles_deg)

    # Define rotation matrices around X, Y, and Z axes
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(angles_rad[0]), -np.sin(angles_rad[0])],
        [0, np.sin(angles_rad[0]), np.cos(angles_rad[0])]
    ])

    rotation_matrix_y = np.array([
        [np.cos(angles_rad[1]), 0, np.sin(angles_rad[1])],
        [0, 1, 0],
        [-np.sin(angles_rad[1]), 0, np.cos(angles_rad[1])]
    ])

    rotation_matrix_z = np.array([
        [np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
        [np.sin(angles_rad[2]), np.cos(angles_rad[2]), 0],
        [0, 0, 1]
    ])

    # Apply rotations around X, Y, and Z axes sequentially
    rotated_points = np.dot(translated_points, rotation_matrix_x.T)
    rotated_points = np.dot(rotated_points, rotation_matrix_y.T)
    rotated_points = np.dot(rotated_points, rotation_matrix_z.T)

    # Apply inverse translation to bring the object back to its original position
    final_points = rotated_points + center
    return final_points


def display_ply_file(ply_file_path, width, height,rotation,counteri,curr_dir,angles_deg,modi):
    # Load the .ply file
    point_cloud_o3d = o3d.io.read_point_cloud(ply_file_path)
    point_cloud_np = np.asarray(point_cloud_o3d.points)
    colors_np = np.asarray(point_cloud_o3d.colors)

    # Define the rotation angle in degrees for the camera (yaw, pitch, roll)
    yaw_deg = angles_deg[0]  # No rotation in yaw (around the up-axis)
    pitch_deg = angles_deg[1]  # Rotate 30 degrees in pitch (up/down)
    roll_deg = angles_deg[2]  # No rotation in roll (around the view-axis)

    # Convert the rotation angles from degrees to radians
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)
    roll_rad = np.radians(roll_deg)

    # Create the rotation matrix for yaw, pitch, and roll
    rotation_matrix = np.array([[np.cos(yaw_rad) * np.cos(pitch_rad),
                                np.cos(yaw_rad) * np.sin(pitch_rad) * np.sin(roll_rad) - np.sin(yaw_rad) * np.cos(roll_rad),
                                np.cos(yaw_rad) * np.sin(pitch_rad) * np.cos(roll_rad) + np.sin(yaw_rad) * np.sin(roll_rad),
                                0],
                                [np.sin(yaw_rad) * np.cos(pitch_rad),
                                np.sin(yaw_rad) * np.sin(pitch_rad) * np.sin(roll_rad) + np.cos(yaw_rad) * np.cos(roll_rad),
                                np.sin(yaw_rad) * np.sin(pitch_rad) * np.cos(roll_rad) - np.cos(yaw_rad) * np.sin(roll_rad),
                                0],
                                [-np.sin(pitch_rad),
                                np.cos(pitch_rad) * np.sin(roll_rad),
                                np.cos(pitch_rad) * np.cos(roll_rad),
                                0],
                                [0, 0, 0, 1]])

    # Apply the rotation to the point cloud
    rotated_point_cloud = np.hstack((point_cloud_np, np.ones((point_cloud_np.shape[0], 1)))) @ rotation_matrix.T

    # Define the camera intrinsic parameters (change as needed)
    focal_length_x = 800
    focal_length_y = 800
    principal_point_x = 400
    principal_point_y = 400

    # Create the 3x3 intrinsic matrix
    intrinsic_matrix = np.array([[focal_length_x, 0, principal_point_x],
                                [0, focal_length_y, principal_point_y],
                                [0, 0, 1]])

    # Plot the rotated point cloud with original color and without axes
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rotated_point_cloud[:, 0], rotated_point_cloud[:, 1], rotated_point_cloud[:, 2], c=colors_np, s=5)  # s=5 for larger point size
    ax.axis('off')  # Turn off axes

    # Save the snapshot with 800x800 pixels
    temp = f"{curr_dir+modi}/rgb/{counteri}.png"
    plt.savefig(temp, dpi=100, bbox_inches='tight', pad_inches=0)

    # Save the 3x3 intrinsic matrix to a text file
    temp = f"{curr_dir+modi}/intrinsics.png"
    np.savetxt(temp, intrinsic_matrix)

    # Save the 4x4 extrinsic matrix to a text file
    temp = f"{curr_dir+modi}/pose/{counteri}.txt"
    np.savetxt(temp, rotation_matrix)

    bbox_min = np.min(rotated_point_cloud, axis=0)
    bbox_max = np.max(rotated_point_cloud, axis=0)
    temp = f"{curr_dir+modi}/bbox.txt"
    temp3 = ' '.join(map(str, bbox_min)) + " " + ' '.join(map(str, bbox_max)) + " " +str(min(bbox_max - bbox_min))     
    text_file = open(temp, "w")
    text_file.write(temp3)
    text_file.close()

    near = np.linalg.norm(bbox_min - bbox_max) / 2.0  # Half of the diagonal distance
    far = near + np.linalg.norm(bbox_max - bbox_min)  # Center plus the diagonal distance
    temp3 = str(near) +" "+ str(far)
    temp = f"{curr_dir+modi}/near_and_far.txt"     
    text_file = open(temp, "w")
    text_file.write(temp3)
    text_file.close()

    plt.close('all')

def read_ply(ply_file_path,curr_scene,modi):
    # Replace with your .ply file path and the desired width and height for display

    os.mkdir(curr_scene+modi)
    os.mkdir(curr_scene+modi+"/pose")
    os.mkdir(curr_scene+modi+"/rgb")
    width = 800
    height = 800

    counteri = 0
    angle_x = range(0,360,36)
    angle_y = range(0,360,36)
    angle_z = range(0,360,90)

    for curr_x in angle_x:
        for curr_y in angle_y:
            for curr_z in angle_z:
                
                display_ply_file(ply_file_path, width, height,None,counteri,curr_scene,[curr_x,curr_y,curr_z],modi)
                counteri +=1

if __name__ == "__main__":
    all_scenes = ["scene0000_00","scene0001_00","scene0002_00","scene0003_00","scene0004_00","scene0005_00"]
    for curr_scene in all_scenes:
        original_ply = f"/home/anas/Desktop/code/ml3d/ScanNet/my_dataset/scans/{curr_scene}/{curr_scene}_vh_clean_2.ply"
        label_ply = f"/home/anas/Desktop/code/ml3d/ScanNet/my_dataset/scans/{curr_scene}/{curr_scene}_vh_clean_2.labels.ply"

        read_ply(original_ply,curr_scene,"_real")
        read_ply(label_ply,curr_scene,"_seg")
