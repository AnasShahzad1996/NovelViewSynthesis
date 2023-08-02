import os
import open3d as o3d
import numpy as np
from PIL import Image


def sample_images_and_poses(ply_file_path, num_samples):
    # Load the .ply file
    pointcloud = o3d.io.read_point_cloud(ply_file_path)

    # Check if the point cloud is empty
    if len(pointcloud.points) == 0:
        raise ValueError("The point cloud is empty.")

    # Shuffle the points to get random samples
    np.random.shuffle(pointcloud.points)

    # Select 'num_samples' points as the sampled points
    sampled_points = pointcloud.points[:num_samples]

    # Sample the camera poses (here, I'm generating random poses for demonstration purposes)
    camera_poses = [np.eye(4) for _ in range(num_samples)]

    return sampled_points, camera_poses

def calculate_intrinsic_matrix(width, height, fov_x_deg, fov_y_deg):
    fov_x_rad = np.radians(fov_x_deg)
    fov_y_rad = np.radians(fov_y_deg)

    focal_length_x = width / (2 * np.tan(fov_x_rad / 2))
    focal_length_y = height / (2 * np.tan(fov_y_rad / 2))

    intrinsic_matrix = np.array([[focal_length_x, 0, width / 2],
                                 [0, focal_length_y, height / 2],
                                 [0, 0, 1]])

    return intrinsic_matrix

def calculate_bounding_box(pointcloud):
    min_bound = pointcloud.get_min_bound()
    max_bound = pointcloud.get_max_bound()
    return min_bound, max_bound


def save_data(sampled_points, camera_poses, width, height, fov_x_deg, fov_y_deg, near, far, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save intrinsics
    intrinsic_matrix = calculate_intrinsic_matrix(width, height, fov_x_deg, fov_y_deg)
    intrinsic_filename = os.path.join(output_dir, "intrinsic.txt")
    np.savetxt(intrinsic_filename, intrinsic_matrix)

    # Save near and far values
    near_far_filename = os.path.join(output_dir, "near_far.txt")
    np.savetxt(near_far_filename, [near, far])

    # Save bounding box
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(sampled_points)
    min_bound, max_bound = calculate_bounding_box(pointcloud)
    bbox_filename = os.path.join(output_dir, "bbox.txt")
    np.savetxt(bbox_filename, np.hstack((min_bound, max_bound)))

    # Create a PointCloud object
    sampled_pointcloud = o3d.geometry.PointCloud()
    sampled_pointcloud.points = o3d.utility.Vector3dVector(sampled_points)

    # Save snapshots of the 3D point cloud from different viewpoints
    for i, pose in enumerate(camera_poses):
        # Create a copy of the original point cloud on CPU to avoid modifying the original one
        pointcloud_copy = sampled_pointcloud

        # Apply the camera pose to the point cloud
        pointcloud_copy.transform(pose)

        # Create a visualizer and add the transformed point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height)
        vis.add_geometry(pointcloud_copy)

        # Set the view control to use the camera pose
        view_control = vis.get_view_control()
        view_control.set_constant_z_near(near)
        view_control.set_constant_z_far(far)

        # Set the camera pose using the lookat method
        lookat = pose[:3, 3]
        up_direction = pose[:3, 1]
        view_control.set_lookat(lookat)
        view_control.set_up(up_direction)

        # Capture a snapshot and save the image
        image_filename = os.path.join(output_dir, f"image_{i}.png")
        vis.capture_screen_image(image_filename, do_render=True)

        # Close the visualizer
        vis.destroy_window()

        # Save the camera pose (extrinsics)
        extrinsic_filename = os.path.join(output_dir, f"extrinsic_{i}.txt")
        np.savetxt(extrinsic_filename, pose)



def save_real_images(ply_file_path,output_dir):
    # Replace with your .ply file path, the desired number of samples, image width, height, field of view angles, near, and far values
    #ply_file_path = "path/to/your/file.ply"
    num_samples = 400
    image_width = 800
    image_height = 800
    fov_x_deg = 60.0
    fov_y_deg = 45.0
    near = 0.1
    far = 100.0

    # Sample images and poses
    sampled_points, camera_poses = sample_images_and_poses(ply_file_path, num_samples)

    # Save data
    save_data(sampled_points, camera_poses, image_width, image_height, fov_x_deg, fov_y_deg, near, far, output_dir)




if __name__ == "__main__":
    all_scenes = ["scene0000_00","scene0001_00","scene0002_00","scene0003_00","scene0004_00","scene0005_00"]
    for curr_scene in all_scenes:
        original_ply = f"/home/anas/Desktop/code/ml3d/ScanNet/my_dataset/scans/{curr_scene}/{curr_scene}_vh_clean_2.ply"
        label_ply = f"/home/anas/Desktop/code/ml3d/ScanNet/my_dataset/scans/{curr_scene}/{curr_scene}_vh_clean_2.labels.ply"

        output1 = f"{curr_scene}_real"
        output2 = f"{curr_scene}_seg"
        save_real_images(original_ply,output1)
        save_real_images(label_ply,output2)