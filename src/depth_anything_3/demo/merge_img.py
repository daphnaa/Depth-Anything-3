import open3d as o3d
redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()

pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
    redwood_rgbd.camera_intrinsic_path)
print(pinhole_camera_intrinsic.intrinsic_matrix)

source_color = o3d.io.read_image(redwood_rgbd.color_paths[0])
source_depth = o3d.io.read_image(redwood_rgbd.depth_paths[0])
target_color = o3d.io.read_image(redwood_rgbd.color_paths[1])
target_depth = o3d.io.read_image(redwood_rgbd.depth_paths[1])
source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    source_color, source_depth)
target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    target_color, target_depth)
target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    target_rgbd_image, pinhole_camera_intrinsic)