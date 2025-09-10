import glob
import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE

imgs_shape = (48, 192, 192, 1)
masks_shape = (48, 192, 192, 4)

def load_npy_to_tf(img_path, mask_path):
    def _load(path):
        path = path.numpy().decode("utf-8")
        return np.load(path).astype(np.float32)

    # set shapes so Keras sees dims
    img = tf.py_function(_load, [img_path], tf.float32)
    mask = tf.py_function(_load, [mask_path], tf.float32)

    img.set_shape(imgs_shape)
    mask.set_shape(masks_shape)

    return img, mask

def load_datasets(dataset_path):
    ### This function assumes the following path structure ###
    # ./dataset_path                                         #
    #             | train                                    #
    #             |     | images                             #
    #             |     | mask                               #
    #             | valid                                    #
    #             |     | images                             #
    #             |     | mask                               #
    ##########################################################

    train_path = "train/"
    val_path = "valid/"
    imgs_path = "images/"
    masks_path = "mask/"

    # Training dataset #
    train_imgs_list = sorted(glob.glob(dataset_path + train_path + imgs_path + "*.npy"))
    train_masks_list = sorted(glob.glob(dataset_path + train_path + masks_path + "*.npy"))
    
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_imgs_list, train_masks_list))
        .map(load_npy_to_tf, num_parallel_calls=AUTOTUNE)
        .shuffle(100)
        .batch(2)
        .prefetch(AUTOTUNE)
    )

    print(f'Train dataset loaded with size {len(train_ds)}')

    # Validation dataset #
    val_imgs_list = sorted(glob.glob(dataset_path + val_path + imgs_path + "*.npy"))
    val_masks_list = sorted(glob.glob(dataset_path + val_path + masks_path + "*.npy"))

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_imgs_list, val_masks_list))
        .map(load_npy_to_tf, num_parallel_calls=AUTOTUNE)
        .batch(2)
        .prefetch(AUTOTUNE)
    )

    print(f'Validation dataset loaded with size {len(val_ds)}')
    
    return train_ds, val_ds