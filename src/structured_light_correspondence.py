import numpy as np
import skimage.io
import scipy.io
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

# Input data information
data_directory = "../Experiments/data/Bowl"
pattern_directory = "../Patterns"
index_length = 2
image_ext = "pgm"

# Dimension params
projector_dim = [768, 1024] # num rows, num columns
camera_dim = [2000, 2400] # num rows, num columns

# Reconstruction params
shadow_threshold = 0.0 # cull shadow pixels (choose val 0 to 0.1)
median_filter_param = [3, 3] # size of filter for median filtering
diff_threshold = 3 # threshold for which decoded vals are considered the same
hole_fill_flag = True # decide if hole filling should occur
median_filter_hole_fill = [17, 17] # size of neighborhood for hole filling, typical values: [9, 9] -> [25, 25]

# Other params
eps = 0.001

def binary_decode(image_indices, directory_name, permcol):
  IC = np.zeros(camera_dim)
  IDiff1 = np.zeros(camera_dim)
  num_col_images = len(image_indices)
  for i in range(num_col_images):
    image_num1 = ("0" + str(image_indices[i]))[-index_length:]
    image_num2 = ("0" + str(image_indices[i] + 1))[-index_length:]
    I1 = skimage.io.imread(f"{directory_name}/{image_num1}.{image_ext}").astype(np.double)
    I2 = skimage.io.imread(f"{directory_name}/{image_num2}.{image_ext}").astype(np.double)
    I_on = I1 > I2

    IC = IC + I_on * (2 ** (num_col_images - i - 1))

    IDiff1 = IDiff1 + np.abs(I1 - I2) / (I1 + I2 + eps)

    bit_plane = 255 * I_on.astype("uint8")
    bit_plane_num = ("0" + str(i + 1))[-2:]
    skimage.io.imsave(f"{directory_name}/BitPlane{bit_plane_num}.bmp", bit_plane)

  # Use permutation to get actual indices
  print("Recovering actual indices from permutation")
  IC = IC + 1
  for i in range(camera_dim[0]):
    for j in range(camera_dim[1]):
      indices = np.where(permcol == IC[i][j])[1]
      if (len(indices) == 0):
        IC[i][j] = 0
      else:
        IC[i][j] = indices[0] + 1
        

  IC = IC - 0.5 
  IDiff1 = IDiff1 / num_col_images

  return IC, IDiff1

def gray_decode(name):
  gray_data_dir = f"{data_directory}/{name}"
  gray_pattern_dir = f"{pattern_directory}/{name}"
  permcol = scipy.io.loadmat(f"{gray_pattern_dir}/permcol.mat")["permcol"]
  image_indices = np.arange(1, 19 + 1, 2) # every other pattern, because inverses are included
  IC_gray, IDiff1_gray = binary_decode(image_indices, gray_data_dir, permcol)
  IC_gray = IC_gray * (IC_gray >= shadow_threshold).astype(int) # eliminate shadow pixels
  IC_gray = median_filter(IC_gray, size=median_filter_param)
  plt.imshow(IC_gray)
  plt.show()

  return IC_gray, IDiff1_gray

def binary_decode_xor(image_indices, image_base_indices, directory_name, permcol):
  num_col_images = len(image_indices)
  image_num1 = ("0" + str(image_base_indices[0]))[-index_length:]
  image_num2 = ("0" + str(image_base_indices[1]))[-index_length:]
  I1 = skimage.io.imread(f"{directory_name}/{image_num1}.{image_ext}")
  I2 = skimage.io.imread(f"{directory_name}/{image_num2}.{image_ext}")
  I_base_on = I1 > I2

  IC = np.zeros(camera_dim)
  IDiff1 = np.zeros(camera_dim)

  for i in range(num_col_images):
    image_num1 = ("0" + str(image_indices[i]))[-index_length:]
    image_num2 = ("0" + str(image_indices[i] + 1))[-index_length:]
    I1 = skimage.io.imread(f"{directory_name}/{image_num1}.{image_ext}")
    I2 = skimage.io.imread(f"{directory_name}/{image_num2}.{image_ext}")
    I_on = I1 > I2

    if (image_indices[i] < image_base_indices[1]):
      I_on = np.bitwise_xor(I_on, I_base_on)

    IC = IC + I_on.astype(int) * (2 ** (num_col_images - i - 1))
    IDiff1 = IDiff1 + np.abs(I1 - I2) / (I1 + I2 + eps)

    bit_plane = 255 * I_on.astype("uint8")
    bit_plane_num = ("0" + str(i + 1))[-2:]
    skimage.io.imsave(f"{directory_name}/BitPlane{bit_plane_num}.bmp", bit_plane)

  # Use permutation to get actual indices
  print("Recovering actual indices from permutation")
  IC = IC + 1
  for i in range(camera_dim[0]):
    for j in range(camera_dim[1]):
      indices = np.where(permcol == IC[i][j])[1]
      if (len(indices) == 0):
        IC[i][j] = 0
      else:
        IC[i][j] = indices[0] + 1
        

  IC = IC - 0.5 
  IDiff1 = IDiff1 / num_col_images
  return IC, IDiff1

def xor_decode(name):
  xor_data_dir = f"{data_directory}/{name}"
  xor_pattern_dir = f"{pattern_directory}/{name}"
  permcol = scipy.io.loadmat(f"{xor_pattern_dir}/permcol.mat")["permcol"]
  image_indices = np.arange(1, 19 + 1, 2)
  image_base_indices = np.array([17, 18])
  IC_xor, IDiff1_xor = binary_decode_xor(image_indices, image_base_indices, xor_data_dir, permcol)
  plt.imshow(IC_xor)
  plt.show()

  return IC_xor, IDiff1_xor

def combined_decode():
  # Calculate separate decodings
  IC_cgray, IDiff1_cgray = gray_decode("ConventionalGray")
  IC_mgray, IDiff1_mgray = gray_decode("MaxMinSWGray")
  IC_xor2, IDiff1_xor2 = xor_decode("XOR02")
  IC_xor4, IDiff1_xor4 = xor_decode("XOR04")

  # Compare decodings
  cgray_mgray = np.abs(IC_cgray - IC_mgray) <= diff_threshold
  cgray_xor2 = np.abs(IC_cgray - IC_xor2) <= diff_threshold
  cgray_xor4 = np.abs(IC_cgray - IC_xor4) <= diff_threshold
  mgray_xor2 = np.abs(IC_mgray - IC_xor2) <= diff_threshold
  mgray_xor4 = np.abs(IC_mgray - IC_xor4) <= diff_threshold
  xor2_xor4 = np.abs(IC_xor2 - IC_xor4) <= diff_threshold

  # Create mask
  IC_mask = np.zeros(IC_cgray.shape)
  IC_mask[np.where(xor2_xor4)] = 1
  IC_mask[np.where(cgray_mgray)] = 2
  IC_mask[np.where(mgray_xor2)] = 3
  IC_mask[np.where(cgray_xor2)] = 4
  IC_mask[np.where(mgray_xor4)] = 5
  IC_mask[np.where(cgray_xor4)] = 6
  IC_mask[np.where(cgray_mgray & cgray_xor4 & cgray_xor2 & mgray_xor4 & mgray_xor2 & xor2_xor4)] = 7

  # Combine decodings using mask
  IC = np.zeros(IC_cgray.shape)
  IC[np.where(IC_mask == 0)] = 0
  IC[np.where(IC_mask == 1)] = IC_xor4[np.where(IC_mask == 1)]
  IC[np.where(IC_mask == 2)] = IC_mgray[np.where(IC_mask == 2)]
  IC[np.where(IC_mask == 3)] = IC_mgray[np.where(IC_mask == 3)]
  IC[np.where(IC_mask == 4)] = IC_xor2[np.where(IC_mask == 4)]
  IC[np.where(IC_mask == 5)] = IC_mgray[np.where(IC_mask == 5)]
  IC[np.where(IC_mask == 6)] = IC_xor4[np.where(IC_mask == 6)]
  IC[np.where(IC_mask == 7)] = IC_mgray[np.where(IC_mask == 7)]

  # Fill holes
  if (hole_fill_flag):
    IC_filt = median_filter(IC, size=median_filter_hole_fill)
    cgray_diff = np.abs(IC_filt - IC_cgray);
    mgray_diff = np.abs(IC_filt - IC_mgray);
    xor4_diff = np.abs(IC_filt - IC_xor4);
    xor2_diff = np.abs(IC_filt - IC_xor2);

    diff = np.dstack([cgray_diff, mgray_diff, xor4_diff, xor2_diff])
    I = np.min(diff, axis=-1)
    IC_hole_filled = IC
    
    IC_hole_filled[np.where((I == 1) & (IC_mask == 0))] = IC_cgray[np.where((I == 1) & (IC_mask == 0))]
    IC_hole_filled[np.where((I == 2) & (IC_mask == 0))] = IC_mgray[np.where((I == 2) & (IC_mask == 0))]
    IC_hole_filled[np.where((I == 3) & (IC_mask == 0))] = IC_xor4[np.where((I == 3) & (IC_mask == 0))]
    IC_hole_filled[np.where((I == 4) & (IC_mask == 0))] = IC_xor2[np.where((I == 4) & (IC_mask == 0))]

    IC = IC_hole_filled
  
  np.savez(f"{data_directory}/DecodedIndices.npz", IC)
  plt.imshow(IC)
  plt.show()
  return IC
  


def main():
  combined_decode()
  return

if __name__ == "__main__":
  main()