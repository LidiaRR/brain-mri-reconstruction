import zipfile
import numpy as np
import nibabel as nib
import glob
import os

### Paths ###
zip_file = "ibrs_dataset.zip"
dataset_path = "ibsr_3d/"
train_path = "train/"
valid_path = "valid/"
imgs_path = "images/"
mask_path = "mask/"
nii_path = "nii_imgs/"

### Unzip dataset ###
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall()

print("Dataset Unzipped")

### Transform images to .nii.gz for external tools ###
voxel_size = (1.0, 3.0, 1.0) 
affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1])

def img_load_transform_export(img_path):
    my_img = np.load(dataset_path + img_path)

    if imgs_path in img_path:
        my_img = np.squeeze(my_img)
    else:
        my_img = np.argmax(my_img, axis=-1).astype(np.uint8)

    my_img = np.transpose(my_img, (2, 0, 1))
    my_img = np.flip(my_img, axis=1)
    my_img = np.flip(my_img, axis=2)

    nii_img = nib.Nifti1Image(my_img, affine)
    nib.save(nii_img, nii_path + img_path[:-3] + "nii.gz")

paths_1 = [train_path, valid_path]
paths_2 = [imgs_path, mask_path]

for path1 in  paths_1:
    for path2 in paths_2:
        if not os.path.exists(nii_path + path1 + path2):
            os.makedirs(nii_path + path1 + path2)

        for path in glob.glob(dataset_path + path1 + path2 + "*.npy"):
            path = path.replace("\\", "/")
            img_load_transform_export(path[len(dataset_path):])

print("Images transformed to nifti format")