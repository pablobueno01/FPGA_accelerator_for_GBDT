#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import sys
import time
import tensorflow as tf
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from inference_reduced import *

# PRINT CALLBACK FUNCTION
# =============================================================================

class _PrintCallback(tf.keras.callbacks.Callback):
    """Callback to print time, loss and accuracy logs during training
    
    Callbacks can be passed to keras methods such as `fit`, `evaluate`,
    and `predict` in order to hook into the various stages of the model
    training and inference lifecycle.
    
    Attributes
    ----------
    print_epoch : int
        The log messages are written each `print_epoch` epochs.
    losses_avg_no : int
        The current loss value is calculated as the average of the last
        `losses_avg_no` batches loss values.
    start_epoch : int
        Number of the initial epoch in case of finetuning.
    
    Methods
    -------
    print_loss_acc(self, logs, time, last=False)
        Prints log messages with time, loss and accuracy values.
    on_train_begin(self, logs={})
        Called at the beginning of training. Instantiates and
        initialises the `losses`, `epoch` and `start_time` attributes.
    on_batch_end(self, batch, logs={})
        Called at the end of a training batch in `fit` methods.
        Actualises the `losses` attribute with the current value of the
        `loss` item in `logs` dict.
    on_epoch_end(self, epoch, logs={})
        Called at the end of an epoch. Actualises epoch counter and
        prints log message on printable epochs.
    on_train_end(self, logs={})
        Called at the end of training. Prints end of training log
        message.
    """
    
    def __init__(self, print_epoch=1000, losses_avg_no=100, start_epoch=0):
        """Inits PrintCallback instance
        
        Parameters
        ----------
        print_epoch : int, optional (default: 1000)
            The log messages are written each `print_epoch` epochs.
        losses_avg_no : int, optional (default: 100)
            The current loss value is calculated as the average of the
            last `losses_avg_no` batches loss values.
        start_epoch : int, optional (default: 0)
            Number of the initial epoch in case of finetuning.
        """
        self.print_epoch = print_epoch
        self.losses_avg_no = losses_avg_no
        self.start_epoch = start_epoch
    
    def print_loss_acc(self, logs, time, last=False):
        """Prints log messages with time, loss and accuracy values
        
        Parameters
        ----------
        logs : dict
            Aggregated metric results up until this batch.
        time : int
            Current training time in seconds.
        last : bool, optional (default: False)
            Flag to activate end of training log message.
        """
        
        # Calculate current loss value
        loss = sum(self.losses[-self.losses_avg_no:])/self.losses_avg_no
        
        # Print log message
        if last:
            print("\n--- TRAIN END AT EPOCH {} ---".format(self.epoch))
            print("TRAINING TIME: {} seconds".format(time))
            end = "\n"
        else:
            print("\nCURRENT TIME: {} seconds".format(time))
            end = ''
        print("Epoch loss ({}): {}".format(self.epoch, loss))
        print("Accuracy: {}".format(logs.get('val_accuracy')), end=end, flush=True)
    
    def on_train_begin(self, logs={}):
        """Called at the beginning of training
        
        Instantiates and initialises the `losses`, `epoch` and
        `start_time` attributes. The `logs` parameter is not used, but
        this is an overwritten method, so it is mandatory.
        
        Parameters
        ----------
        logs : dict
            Currently no data is passed to this argument for this
            method but that may change in the future.
        """
        self.losses = []
        self.epoch = self.start_epoch
        self.start_time = time.time()
    
    def on_batch_end(self, batch, logs={}):
        """Called at the end of a training batch in `fit` methods
        
        Actualises the `losses` attribute with the current value of the
        `loss` item in `logs` dict. The `batch` parameter is not used,
        but this is an overwritten method, so it is mandatory.
        
        This is a backwards compatibility alias for the current method
        `on_train_batch_end`.
        
        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called
        every `N` batches.
        
        Parameters
        ----------
        batch : int
            Index of batch within the current epoch.
        logs : dict
            Aggregated metric results up until this batch.
        """
        self.losses.append(logs.get('loss'))
    
    def on_epoch_end(self, epoch, logs={}):
        """Called at the end of an epoch
        
        Actualises epoch counter and prints log message on printable
        epochs.
        
        This function should only be called during TRAIN mode.
        
        Parameters
        ----------
        epoch : int
            Index of epoch.
        logs : dict
            Metric results for this training epoch, and for the
            validation epoch if validation is performed. Validation
            result keys are prefixed with `val_`. For training epoch,
            the values of the `Model`'s metrics are returned.
        """
        
        # Actualise epoch
        self.epoch += 1
        
        # If it is a printable epoch
        if self.epoch % self.print_epoch == 0:
            
            # Print log message
            current_time = time.time() - self.start_time
            self.print_loss_acc(logs, current_time)
    
    def on_train_end(self, logs={}):
        """Called at the end of training
        
        Prints end of training log message.
        
        Parameters
        ----------
        logs : dict
            Currently the output of the last call to `on_epoch_end()`
            is passed to this argument for this method but that may
            change in the future.
        """
        total_time = time.time() - self.start_time
        self.print_loss_acc(logs, total_time, last=True)

# MODEL FUNCTION
# =============================================================================

def get_model(dataset_size, num_features, num_classes, l1_n, l2_n,
              learning_rate):
    """Generates the bayesian model
    
    Parameters
    ----------
    dataset_size : int
        Number of pixels of the dataset.
    num_features : int
        Number of features of each pixel.
    num_classes : int
        Number of classes of the dataset.
    l1_n : int
        Number of neurons of the first hidden layer.
    l2_n : int
        Number of neurons of the second hidden layer
    learning_rate : float
        Initial learning rate.
    
    Returns
    -------
    model : TensorFlow Keras Sequential
        Bayesian model ready to receive and train hyperspectral data.
    """
    
    # Generate and compile model
    tf.keras.backend.clear_session()
    kd_function = (lambda q, p, _: dist.kl_divergence(q, p)/dataset_size)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(num_features,), name="input"),
        tfp.layers.DenseFlipout(l1_n, kernel_divergence_fn=kd_function,
                                activation=tf.nn.relu, name="dense_tfp_1"),
        tfp.layers.DenseFlipout(l2_n, kernel_divergence_fn=kd_function,
                                activation=tf.nn.relu, name="dense_tfp_2"),
        tfp.layers.DenseFlipout(num_classes,
                                kernel_divergence_fn=kd_function,
                                activation=tf.nn.softmax, name="output"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# MAIN FUNCTION
# =============================================================================

def train(name, epochs, period, l1_n=32, l2_n=16, p_train=0.15, learning_rate=1.0e-2):
    """Trains a bayesian model for a hyperspectral image dataset
    
    The trained model and the checkouts are saved in the `MODELS_DIR`
    defined in `config.py`.
    
    Parameters
    ----------
    name : str
        Abbreviated name of the dataset.
    epochs : int
        Total number of epochs to train.
    period : int
        Checkpoints and information period.
    mix_classes : bool, optional (default: False)
        Flag to activate mixed classes training.
    """
    
    # Input, output and dataset references
    base_output_dir = "./bnn_results/models"
    datasets = IMAGES
    
    # DATASET INFORMATION
    # -------------------------------------------------------------------------
    
    dataset = datasets[name]
    
    # Extract dataset classes and features
    num_classes = dataset['num_classes']
    num_features = dataset['num_features']
    
    # Generate output dir
    output_dir = "{}_{}-{}model_{}train_{}lr".format(name, l1_n, l2_n, p_train, learning_rate)
    output_dir = os.path.join(base_output_dir, output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # GET DATA
    # -------------------------------------------------------------------------

    # Load image
    X, y = load_image(dataset)
    
    # Preprocess image
    X, y = pixel_classification_preprocessing(X, y)
    
    # Separate data into train and test sets
    X_train, y_train, X_test, _ = separate_pixels(X, y, dataset["p"])
    print("Train pixels: {}\tTest pixels: {}".format(X_train.shape[0], X_test.shape[0]))
    
    # TRAIN MODEL
    # -------------------------------------------------------------------------
    
    # Get model (if already trained, continue for finetuning)
    trained = [int(d.split("_")[1]) for d in os.listdir(output_dir)
               if "_" in d]
    if trained:
        initial_epoch = max(trained)
        last_file = os.path.join(output_dir, "epoch_{}".format(initial_epoch))
        model = tf.keras.models.load_model(last_file)
    else:
        initial_epoch = 0
        dataset_size = len(X_train) + len(X_test)
        model = get_model(dataset_size, num_features, num_classes, l1_n, l2_n,
                          learning_rate)
        
    # PRINT CALLBACK
    print_callback = _PrintCallback(print_epoch=period,
                                    losses_avg_no=max(1, period//10),
                                    start_epoch=initial_epoch)
    
    # CHECKPOINT CALLBACK
    file = os.path.join(output_dir, "epoch_{epoch}")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file,
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    mode='max',
                                                    save_best_only=False,
                                                    period=period)
    
    # Print start training message
    msg = "\n### Starting the {} training on epoch {}"
    print(msg.format(name, initial_epoch))
    print('#'*80)
    print("\nOUTPUT DIR: {}".format(output_dir))
    
    # Training
    model.fit(X_train,
              tf.one_hot(y_train, num_classes),
              initial_epoch=initial_epoch,
              epochs=epochs, 
              verbose=0,
              use_multiprocessing=True,
              callbacks=[print_callback, checkpoint],
              validation_split=0.1,
              validation_freq=25)
    
    # Save model
    model.save(os.path.join(output_dir, "final"))

if __name__ == "__main__":
    
    for image in IMAGES.keys():
        train(image, 10000, 1000)