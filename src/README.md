# Structured Light Correspondence
Set the parameters in `./src/structured_light_correspondence.py`:
```
data_directory = "../data/flour" # the directory for captured images
pattern_directory = "../Patterns" # patterns directory
index_length = 2 # how many digits for image name
image_ext = "JPG" # the extension for the images

projector_dim = [768, 1024] # num rows, num columns
camera_dim = [667, 1000] # num rows, num columns
```

Note that the directory chosen as `data_directory` should contain four subfolders: `ConventionalGray`, `MaxMinSWGray`, `XOR02`, and `XOR04`, with images in each subfolder labeled from 01 to 20. 

The ordering of the images must match the ordering of the patterns given in `pattern_directory`. The `pattern_directory` must also contain the same four subfolders as `data_directory`. 

Once the parameters are set, run `py structured_light_correspondence.py` while located in the `src` folder.

# Calibration
Calibration pictures are given in `./data/calib-cam` and `./data/calib-proj`. If you want to use your own calibration files, change the call to the function `proj_cam_calib` inside of `./data/calibrate.py`.

To run, you should run `py calibrate.py` from inside the `src` file. This is meant to perform camera-projector calibration.