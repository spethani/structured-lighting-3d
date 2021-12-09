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
    image_num1 = ("0" + str(image_indices[i]))[-2:]
    image_num2 = ("0" + str(image_indices[i] + 1))[-2:]
    I1 = skimage.io.imread(f"{directory_name}/{image_num1}.{image_ext}")
    I2 = skimage.io.imread(f"{directory_name}/{image_num2}.{image_ext}")
    I_on = I1 > I2

    IC = IC + I_on.astype(int) * (2 ** (num_col_images - i - 1))
    IDiff1 = IDiff1 + np.abs(I1 - I2) / (I1 + I2 + eps)

    bit_plane = 255 * I_on.astype("uint8")
    bit_plane_num = ("0" + str(i + 1))[-2:]
    skimage.io.imsave(f"{directory_name}/BitPlane{bit_plane_num}.bmp", bit_plane)

  print(IC)
  # Use permutation to get actual indices
  print("Recovering actual indices from permutation")
  for i in range(camera_dim[0]):
    for j in range(camera_dim[1]):
      indices = np.where(permcol == IC[i][j])[1]
      if (len(indices) == 0):
        IC[i][j] = 0
      else:
        IC[i][j] = indices[0]
        

  IC = IC - 0.5 
  IDiff1 = IDiff1 / num_col_images

  return IC, IDiff1

def gray_decode(name):
  gray_data_dir = f"{data_directory}/{name}"
  gray_pattern_dir = f"{pattern_directory}/{name}"
  permcol = scipy.io.loadmat(f"{gray_pattern_dir}/permcol.mat")["permcol"]
  image_indices = np.arange(1, 19, 2) # every other pattern, because inverses are included
  IC_gray, IDiff1_gray = binary_decode(image_indices, gray_data_dir, permcol)
  IC_gray = IC_gray * (IC_gray >= shadow_threshold).astype(int) # eliminate shadow pixels
  IC_gray = median_filter(IC_gray, size=median_filter_param)
  plt.imshow(IC_gray)
  plt.show()

  return IC_gray, IDiff1_gray


def combined_decode():
  # Conventional Gray
  gray_decode("ConventionalGray")

  # Max Min SW Gray
  gray_decode("MaxMinSWGray")

def main():
  combined_decode()
  return

if __name__ == "__main__":
  main()