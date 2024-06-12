#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Noise testing module of the bnn4hi package

This module contains the main function to generate the `combined noise`
plot of the trained bayesian models.

This module can be imported as a part of the bnn4hi package, but it can
also be launched from command line, as a script. For that, use the `-h`
option to see the required arguments.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Local imports
if '.' in __name__:
    
    # To run as a module
    from .lib import config
    from .lib.data import get_noisy_dataset
    from .lib.model import get_model
    from .lib.analysis import bayesian_predictions, analyse_entropy
    from .lib.plot import plot_combined_noise

else:
    
    # To run as an script
    from lib import config
    from lib.data import get_noisy_dataset
    from lib.model import get_model
    from lib.analysis import bayesian_predictions, analyse_entropy
    from lib.plot import plot_combined_noise

# Testing all the images and generating all the noisy data can generate GPU
# memory errors.
# Try to comment this line if you have a big GPU. In any case, it will save the
# result of each dataset for future executions in case there are memory errors.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# PARAMETERS
# =============================================================================

def _parse_args(dataset_list):
    """Analyses the received parameters and returns them organised.
    
    Takes the list of strings received at sys.argv and generates a
    namespace assigning them to objects.
    
    Parameters
    ----------
    dataset_list : list of str
        List with the abbreviated names of the datasets to test. If
        `test_noise.py` is launched as a script, the received
        parameters must correspond to the order of this list.
    
    Returns
    -------
    out : namespace
        The namespace with the values of the received parameters
        assigned to objects.
    """
    
    # Generate the parameter analyser
    parser = ArgumentParser(description = __doc__,
                            formatter_class = RawDescriptionHelpFormatter)
    
    # Add arguments
    parser.add_argument("epochs",
                        type=int,
                        nargs=len(dataset_list),
                        help=("List of the epoch of the selected checkpoint "
                              "for testing each model. The order must "
                              f"correspond to: {dataset_list}."))
    
    # Return the analysed parameters
    return parser.parse_args()

# PREDICT FUNCTION
# =============================================================================

def noise_predict(model, X_test, y_test, samples=100):
    """Launches the bayesian noise predictions
    
    Launches the necessary predictions over `model` to collect the data
    to generate the `combined noise` plot.
    
    Parameters
    ----------
    model : TensorFlow Keras Sequential
        The trained model.
    X_test : ndarray
        Testing data set.
    y_test : ndarray
        Testing data set labels.
    samples : int, optional (default: 100)
        Number of bayesian passes to perform.
    
    Returns
    -------
    avg_H : ndarray
        List of the averages of the global uncertainty (H) of each
        class. The last position also contains the average of the
        entire image.
    """
    
    # Bayesian stochastic passes
    print(f"\nLaunching {samples} bayesian predictions")
    predictions = bayesian_predictions(model, X_test, samples=samples)
    
    # Analyse entropy
    avg_H, _, _ = analyse_entropy(predictions, y_test)
    
    return avg_H

# MAIN FUNCTION
# =============================================================================

def test_noise(epochs):
    """Generates the `combined noise` plot of the trained models
    
    The plot is saved in the `TEST_DIR` defined in `config.py`.
    
    Parameters
    ----------
    epochs : dict
        Dict structure with the epochs of the selected checkpoint for
        testing each model. The keys must correspond to the abbreviated
        name of the dataset of each trained model.
    """
    
    # CONFIGURATION MACROS (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------
    
    # Input, output and dataset references
    d_path = config.DATA_PATH
    base_dir = config.MODELS_DIR
    datasets = config.DATASETS
    output_dir = config.TEST_DIR
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Model parameters
    l1_n = config.LAYER1_NEURONS
    l2_n = config.LAYER2_NEURONS
    
    # Training parameters
    p_train = config.P_TRAIN
    learning_rate = config.LEARNING_RATE
    
    # Noise testing parameters
    noises = config.NOISES
    
    # Bayesian passes
    passes = config.BAYESIAN_PASSES
    
    # Plot parameters
    colours = config.COLOURS
    w = config.PLOT_W
    h = config.PLOT_H
    
    # Plotting variables
    data = {}
    
    # FOR EVERY DATASET
    # -------------------------------------------------------------------------
    for name, dataset in datasets.items():
        
        # DATASET INFORMATION
        # ---------------------------------------------------------------------
        
        # Extract dataset classes and features
        num_classes = dataset['num_classes']
        num_features = dataset['num_features']
        
        # Get model dir
        model_dir = (f"{name}_{l1_n}-{l2_n}model_{p_train}train"
                     f"_{learning_rate}lr")
        model_dir = os.path.join(model_dir, f"epoch_{epochs[name]}")
        model_dir = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_dir):
            data[name] = []
            continue
        
        # GENERATE OR LOAD NOISY PREDICTIONS
        # ---------------------------------------------------------------------
        
        # If noisy predictions file already exists
        noise_file = os.path.join(model_dir, "test_noise.npy")
        if os.path.isfile(noise_file):
            
            # Load message
            print(f"\n### Loading {name} noise test")
            print('#'*80)
            print(f"\nMODEL DIR: {model_dir}", flush=True)
            
            # Load it
            noise_data = np.load(noise_file)
        
        else:
            
            # GET DATA
            # -----------------------------------------------------------------
            
            # Get noisy datasets
            _, _, n_X_tests, n_y_test = get_noisy_dataset(dataset, d_path,
                                                          p_train, noises)
            
            # LOAD MODEL
            # -----------------------------------------------------------------
            
            # Load model
            model = tf.keras.models.load_model(model_dir)
            
            # LAUNCH PREDICTIONS
            # -----------------------------------------------------------------
            
            # Noise test message
            print(f"\n### Starting {name} noise test")
            print('#'*80)
            print(f"\nMODEL DIR: {model_dir}")
            
            # Launch predictions for every noisy dataset
            noise_data = [[] for i in range(num_classes + 1)]
            for n, n_X_test in enumerate(n_X_tests):
                
                # Test message
                print(f"\n# Noise test {n + 1} of {len(n_X_tests)}")
                
                # Launch prediction
                avg_H = noise_predict(model, n_X_test, n_y_test,
                                      samples=passes)
                noise_data = np.append(noise_data, avg_H[np.newaxis].T, 1)
            
            # Save result
            np.save(os.path.join(model_dir, "test_noise"), noise_data)
            
            # Liberate model
            del model
        
        # Add normalised average to data structure
        max_H = np.log(num_classes)
        data[name] = noise_data[-1]/max_H
    
    # Plot combined noise
    plot_combined_noise(output_dir, noises, data, w, h, colours)

if __name__ == "__main__":
    
    # Parse args
    dataset_list = config.DATASETS_LIST
    args = _parse_args(dataset_list)
    
    # Generate parameter structures for main function
    epochs = {}
    for i, name in enumerate(dataset_list):
        epochs[name] = args.epochs[i]
    
    # Launch main function
    test_noise(epochs)
