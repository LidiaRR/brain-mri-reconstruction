import tensorflow as tf
from tensorflow.keras import layers, models

def downsampling_block(x, filters, kernel_size=3):
    x1 = layers.Conv3D(filters, kernel_size=kernel_size, padding="same", activation="relu")(x)
    x2 = layers.MaxPool3D((2, 2, 2))(x1)

    return x1, x2

def upsampling_block(x_down, x_up, filters, kernel_size=3):
    x_up = layers.UpSampling3D((2, 2, 2))(x_up)
    x = layers.Concatenate(axis=-1)([x_down, x_up])
    x = layers.Conv3D(filters, kernel_size=kernel_size, padding="same", activation="relu")(x)

    return x

def unet_v1(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape) ## (batch_size, 48, 192, 192, 1)

    x1, x = downsampling_block(inputs, 16)
    x2, x = downsampling_block(x, 32)
    x3, x = downsampling_block(x, 64)

    x = layers.Conv3D(128, kernel_size=3, padding="same", activation="relu")(x)

    x = upsampling_block(x3, x, 64)
    x = upsampling_block(x2, x, 32)
    x = upsampling_block(x1, x, 16)

    outputs = layers.Conv3D(num_classes, kernel_size=1, activation="softmax")(x)

    return models.Model(inputs, outputs, name="unet_v1")


    
