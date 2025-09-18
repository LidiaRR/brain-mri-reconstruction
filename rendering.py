import numpy as np
import os
from skimage import measure
import trimesh

def prepare_img_export(img):
    img = np.squeeze(img)
    img = np.argmax(img, axis=-1).astype(np.uint8)

    img = np.transpose(img, (2, 0, 1))
    img = np.flip(img, axis=1)
    img = np.flip(img, axis=2)

    return img

def export_mesh(mask, filename, x_space, y_space, z_space):
    verts, faces, normals, _ = measure.marching_cubes(mask, level=0.5, spacing=(z_space, y_space, x_space))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    mesh.export(filename)
    print(f"Saved {filename}")

slice_filenames = {1: ["left", "right"], # sagittal
                2: ["front", "back"], # coronal
                3: ["bottom", "top"]} # axial

def rendering(segmentation, x_space=1.0, y_space=3.0, z_space=1.0, slice=0):
    print(segmentation.shape)
    if not os.path.exists('blender_files'):
        os.makedirs('blender_files')

    segmentation = prepare_img_export(segmentation)
    halves = np.array(segmentation.shape) // 2 + 1
    print(halves)

    for label, name in {1: "CSF", 2: "GM", 3: "WM"}.items():
        mask = (segmentation == label).astype(np.uint8)
        if slice == 0:
            filename = f"blender_files/brain_{name}.stl"

            export_mesh(mask, filename, x_space, y_space, z_space)
        else:
            if slice == 1:
                mask1 = mask.copy()
                mask1[halves[0]:, :, :] = 0

                mask2 = mask.copy()
                mask2[:halves[0], :, :] = 0
            elif slice == 2:
                mask1 = mask.copy()
                mask1[:, halves[1]:, :] = 0

                mask2 = mask.copy()
                mask2[:, :halves[1], :] = 0
            else:
                mask1 = mask.copy()
                mask1[:, :, halves[2]:] = 0

                mask2 = mask.copy()
                mask2[:, :, :halves[2]] = 0

            filename1 = f"blender_files/brain_{name}_{slice_filenames[slice][0]}.stl"
            export_mesh(mask1, filename1, x_space, y_space, z_space)

            filename2 = f"blender_files/brain_{name}_{slice_filenames[slice][1]}.stl"
            export_mesh(mask2, filename2, x_space, y_space, z_space)