import os
import matplotlib.pyplot as plt
from models.unet_v1 import unet_v1
from train.loss_functions import dice_loss
from tensorflow.keras.callbacks import TensorBoard
from train.callbacks import ImageLogger

def save_loss_graph(history, save_name):
    if not os.path.exists('train/loss_graphs'):
        os.makedirs('train/loss_graphs')

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per epoch')
    plt.legend()
    plt.savefig(f'train/loss_graphs/loss_{save_name}.png')
    plt.close()

def train_model(train_ds, val_ds, save_name='unet_v1_1', imgs_shape=(48, 192, 192, 1), num_classes=4):
    # Train model #
    model = unet_v1(imgs_shape, num_classes)
    model.compile(optimizer='adam', loss=dice_loss)

    if not os.path.exists(f'train/logs/{save_name}'):
        os.makedirs(f'train/logs/{save_name}')
    tensorboard_cb = TensorBoard(log_dir=f'train/logs/{save_name}', histogram_freq=1)

    image_logger_cb = ImageLogger(f'train/logs/{save_name}', val_ds, num_images=3)

    history = model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=[tensorboard_cb, image_logger_cb])
    
    # Save model #
    if not os.path.exists('train/saved_models'):
        os.makedirs('train/saved_models')
    model.save(f'train/saved_models/{save_name}.keras')

    save_loss_graph(history, save_name)

    return model

