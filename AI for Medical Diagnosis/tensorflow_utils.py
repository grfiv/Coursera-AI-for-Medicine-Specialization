"""
tensorflow_utils.py

This module contains utility functions and classes to help with TensorFlow training, model evaluation, and logging.

Features:
- Custom callbacks for Keras training
- Utilities for model performance evaluation
- Time-tracking and logging functionalities
- Printing of loss functions

Last modified:
    October 16, 2024
"""

import pytz
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def combine_features(feature_dict):
    """
    Combine dictionary of feature tensors into a single tensor.
    Excludes columns that should not be included in the feature set.
    Args:
        feature_dict (dict): Dictionary of feature tensors.
    Returns:
        tf.Tensor: Combined tensor with shape (batch_size, num_features).

    Example of use:
    # features_train is a dataframe resulting from train_test_split
    
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(features_train), labels_train))
    train_dataset = train_dataset.map(lambda x, y: (combine_features(x), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)

    Note:
    in this example we want to exclude two variables from this process:
    
    exclude_columns = ['OriginalIndex', 'Base MSRP']
      # 'Base MSRP' is the target variable, the label, y_train
      # 'OriginalIndex' is original indexes before train_test_split, so I can recreate the test dataframe with its pre-normalized values and column names
    """
    # List of columns to exclude
    exclude_columns = ['OriginalIndex', 'Base MSRP']

    # Filter out excluded columns from the feature dictionary
    filtered_features = {k: v for k, v in feature_dict.items() if k not in exclude_columns}

    # Convert all tensors to the same type and combine
    feature_tensors = [tf.cast(tf.expand_dims(v, axis=-1), tf.float32) for v in filtered_features.values()]
    
    return tf.concat(feature_tensors, axis=-1)


def loss_curves(history, start_after=0, losses=['loss', 'rmse', 'mae'], best_epoch=None):
    """
    Plots loss curves for a training history over the specified range of epochs.
    
    Args:
        history (History object): The training history object returned from the `model.fit()` method, 
                                  containing the metrics and losses for each epoch.
        start_after (int, optional): The number of epochs after which to start plotting the curves. 
                                     Defaults to 0 (plot all epochs).
        losses (list, optional): A list of loss/metric names to plot. The first two will be used in the 
                                 first plot (e.g., 'loss' and 'rmse'), and the third in the second plot. 
                                 Defaults to ['loss', 'rmse', 'mae'].
                                 The second entry describes what sort of loss the first entry is
        best_epoch (int, optional): The epoch where the early stopping occurred (or another indicator of 
                                    the "best" epoch). If not provided, no vertical line will be plotted.

    Raises:
        ValueError: If the loss names provided in `losses` do not exist in the history object.
    
    Example:
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[early_stopping])
        loss_curves(history, start_after=10, losses=['loss', 'rmse', 'mae'], best_epoch=early_stopping.best_epoch)
    """
    
    # Extract epochs for plotting
    epochs = list(range(len(history.history['loss'])))
    
    # Validate that the provided losses exist in the history
    for i,loss in enumerate(losses):
        # losses[1] is intended to say what losses[0] is
        if i == 1: continue
        if loss not in history.history or f'val_{loss}' not in history.history:
            raise ValueError(f"Loss '{loss}' not found in history. Check the 'losses' argument.")
    
    loss_1a = losses[0]  # First loss for training
    loss_1b = losses[1]  # First loss for validation
    loss_2  = losses[2]  # Second loss for training/validation
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # First subplot: Training and Validation loss
    ax1.plot(epochs[start_after:], history.history[loss_1a][start_after:], label='Training Loss')
    ax1.plot(epochs[start_after:], history.history['val_' + loss_1a][start_after:], label='Validation Loss')
    
    if start_after > 0:
        ax1.set_title(f'Loss ({loss_1b}) after {start_after} epochs')
    else:
        ax1.set_title(f'Loss ({loss_1b})')

    if best_epoch is not None and best_epoch < len(epochs):
        ax1.axvline(x=best_epoch, color='red', linestyle='--')
        ax1.set_xlabel(f'Epochs (Red line marks the best_epoch = {best_epoch})')
    else:
        ax1.set_xlabel('Epochs')

    ax1.set_ylabel(f'Loss ({loss_1b})')
    ax1.legend()

    # Second subplot: Another metric, typically RMSE or MAE
    ax2.plot(epochs[start_after:], history.history[loss_2][start_after:], label=f'Training {loss_2}')
    ax2.plot(epochs[start_after:], history.history['val_' + loss_2][start_after:], label=f'Validation {loss_2}')
    
    if start_after > 0:
        ax2.set_title(f'{loss_2} after {start_after} epochs')
    else:
        ax2.set_title(f'{loss_2}')
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(f'{loss_2}')
    ax2.legend()

    # Show the plots
    plt.show()

class PrintEveryNEpoch(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to print logs at the end of every N epochs during model training.
    
    Parameters:
    -----------
    n : int
        The frequency (in epochs) at which the logs should be printed.

    timezone_str : str
                   the string of the local timezone
                   
                   Use pytz.all_timezones to get a list of all valid time zone strings
                   Or https://gist.github.com/heyalexej/8bf688fd67d7199be4a1682b3eec7568
                   'America/New_York'
                   'US/Mountain'
                   'Europe/London'
                   'Asia/Tokyo'
                   
    
    Methods:
    --------
    on_epoch_end(epoch, logs=None):
        This method is called at the end of every epoch during training.
        It prints the available metrics (from logs) every N epochs.
    
    Notes:
    ------
    - `logs` is a dictionary containing the metric values such as 'loss', 
      'val_loss', 'accuracy', 'val_accuracy', or any other custom metrics 
      being tracked.
    - This callback dynamically prints all the available keys in the logs 
      dictionary, ensuring it works for any metrics being tracked.
    
    Usage:
    ------
    # Create the callback
    periodic_messages = PrintEveryNEpoch(n=150, timezone_str='America/New_York')
    
    # Include it in model.fit (be sure to say verbose=0)
    model.fit(train_tf, epochs=5000, batch_size=32, 
              validation_data=valid_tf, callbacks=[periodic_messages], verbose=0)
    """
    
    def __init__(self, n, timezone_str='UTC'):
        super(PrintEveryNEpoch, self).__init__()
        self.n = n
        try:
            self.local_tz = pytz.timezone(timezone_str)
        except pytz.UnknownTimeZoneError:
            print(f"Unknown timezone: {timezone_str}. Defaulting to UTC.")
            self.local_tz = pytz.utc

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}  # Ensure logs is a dictionary if None
            
        # Every 'n' epochs, print the metrics
        if ((epoch + 1) % self.n == 0) or (epoch == 0):
            # Get the current time in the local timezone
            utc_now    = datetime.datetime.now(pytz.utc)
            local_time = utc_now.astimezone(self.local_tz).strftime('%Y-%m-%d %H:%M:%S')

            # format the log string
            log_str = f"\nEpoch {epoch + 1} ({local_time}): \n"
            for key, value in logs.items():
                log_str += f"{key} = {value:.4f}, "
            # Remove the trailing comma and space
            log_str = log_str.rstrip(', ')
            print(log_str)

