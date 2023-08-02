from open3d import *    

# Load the .sens file
mid = "scene0002_00"
sens_file_path = f"/home/anas/Desktop/code/ml3d/ScanNet/my_dataset/scans/{mid}/{mid}_vh_clean_2.labels.ply"

def main():
    cloud = io.read_point_cloud(sens_file_path) # Read point cloud
    visualization.draw_geometries([cloud])    # Visualize point cloud      

if __name__ == "__main__":
    main()
