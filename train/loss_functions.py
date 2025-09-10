import tensorflow as tf

def dice_loss(y_true, y_pred, smooth=10e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[0,1,2,3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[0,1,2,3])

    dice = (2. * intersection + smooth) / (denominator + smooth)

    return 1 - tf.reduce_mean(dice)