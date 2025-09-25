import tensorflow as tf
import numpy as np

class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, val_data, num_images=3):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir + "/images")
        self.val_data = val_data.take(1)
        self.num_images = num_images
        
    def on_epoch_end(self, epoch, logs=None):
        for images, masks in self.val_data:
            preds = self.model.predict(images)

            images = images[:self.num_images]
            masks = masks[:self.num_images]
            preds = preds[:self.num_images]

            slice_idx = images.shape[1] // 2
            images_slices = images[:, slice_idx, :, :, :]
            masks_slices = masks[:, slice_idx, :, :, :]
            preds_slices = preds[:, slice_idx, :, :, :]

            if masks_slices.shape[-1] > 1:
                masks_slices = tf.cast(tf.expand_dims(tf.argmax(masks_slices, axis=-1), axis=-1), tf.float32)
            if preds_slices.shape[-1] > 1:
                preds_slices = tf.cast(tf.expand_dims(tf.argmax(preds_slices, axis=-1), axis=-1), tf.float32)

            images_slices = (images_slices - tf.reduce_min(images_slices)) / (tf.reduce_max(images_slices) - tf.reduce_min(images_slices) + 1e-6)
            masks_slices = masks_slices / (tf.reduce_max(masks_slices) + 1e-6)
            preds_slices = preds_slices / (tf.reduce_max(preds_slices) + 1e-6)

            with self.file_writer.as_default():
                tf.summary.image("Input", images_slices, step=epoch)
                tf.summary.image("Ground Truth", masks_slices, step=epoch)
                tf.summary.image("Prediction", preds_slices, step=epoch)
            break