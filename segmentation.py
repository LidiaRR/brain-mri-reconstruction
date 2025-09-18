import numpy as np
import tensorflow as tf
from train.loss_functions import dice_loss

def segment(file):
    img = np.load(file)
    img = np.expand_dims(img, axis=0)

    model = tf.keras.models.load_model('train/saved_models/unet_v1_1.keras',
                                       custom_objects={"dice_loss": dice_loss})
    return model(img)