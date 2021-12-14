from cp_hw6 import computeIntrinsic, computeExtrinsic, pixel2ray
import os
import glob
import numpy as np
import cv2

image_ext = "JPG"
dW1 = (8, 8) # window size for finding checkerboard corners

dX = 558.8 # calibration plane length in x direction
dY = 303.2125 # calibration plane length in y direction
dW2 = (8, 8) # window size finding ground plane corners

def intersect_ray_and_plane(ray, points):
  # Assuming points and ray are in same coord system
  camera_p0 = points[0]
  camera_p1 = points[1]
  camera_p2 = points[2]

  # Calculate intersection of plane and ray
  p01 = (camera_p1 - camera_p0).flatten()
  p02 = (camera_p2 - camera_p0).flatten()

  t = np.dot(np.cross(p01, p02), -1 * camera_p0)
  t = t / np.dot(-1 * ray, np.cross(p01, p02))
  intersect = ray * t

  return intersect.flatten()

def proj_cam_calib(cam_checkerboard, cam_dir, proj_checkerboard, proj_dir):
  # Find intrinsics for camera
  cam_calib_images = glob.glob(os.path.join(cam_dir, "*"+image_ext))
  cam_mtx, cam_dist = computeIntrinsic(cam_calib_images, cam_checkerboard, dW1)

  # For each projector image:
  proj_calib_images = glob.glob(os.path.join(proj_dir, "*"+image_ext))
  tvec_projs = []
  rmat_projs = []
  proj_corners = []
  proj_corners_3d = []
  img_shape = None
  for fname in proj_calib_images:
    # find extrinsics
    tvec_proj, rmat_proj = computeExtrinsic(fname, cam_mtx, cam_dist, dX, dY)
    tvec_projs.append(tvec_proj)
    rmat_projs.append(rmat_proj)
    # find checkerboard corners
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, proj_checkerboard, None)
    if (not ret):
      print("Failed to recognize checkerboard corners.")
      continue
    
    print("Found number of checkerboard squares needed.")
    img = cv2.drawChessboardCorners(img, proj_checkerboard, corners, ret)
    cv2.imshow('img',img)
    cv2.waitKey(0)

    proj_corners.append(corners)
    corners = corners.reshape((corners.shape[0], corners.shape[2]))

    # transform corners to camera
    ones = np.ones((corners.shape[0], 1))
    corners_homogenous = np.hstack((corners, ones))
    corners_homogenous = np.transpose(corners_homogenous)
    transformed_corners = np.matmul(rmat_proj, corners_homogenous)
    transformed_corners = transformed_corners + tvec_proj
    transformed_corners = transformed_corners.transpose()

    # for each checkerboard corner:
    corners_3d = []
    for corner in corners:
      # project pixel into ray using camera intrinsics
      pixel_ray = pixel2ray(corner, cam_mtx, cam_dist)
      # perform ray-plane intersection to retrieve 3D point
      # plane used is from transformed corners
      intersect = intersect_ray_and_plane(pixel_ray, transformed_corners)
      corners_3d.append(intersect)
    proj_corners_3d.append(np.array([corners_3d]).astype("float32"))

  print(len(proj_corners_3d), proj_corners_3d[0].shape)
  print(len(proj_corners), proj_corners[0].shape)

  # use 2d points for projector and 3d points to perform calibration
  ret, proj_mtx, proj_dist, rvecs, tvecs = cv2.calibrateCamera(proj_corners_3d, proj_corners, img_shape, cam_mtx, cam_dist, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

  return cam_mtx, cam_dist, proj_mtx, proj_dist

def main():
  cam_mtx, cam_dist, proj_mtx, proj_dist = proj_cam_calib((6, 8), "../data/calib-cam-lr", (7, 7), "../data/calib-proj-lr")
  

if __name__ == "__main__":
  main()

