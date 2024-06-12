#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mixed classes testing module of the bnn4hi package

This module contains the main function to generate the `mixed classes`
table of the `mixed classes` trained bayesian models. It also generates
an individual `mixed classes` plot for each model.

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
    from .lib.data import get_dataset, get_mixed_dataset
    from .lib.model import get_model
    from .lib.analysis import bayesian_predictions, analyse_entropy
    from .lib.plot import plot_mixed_uncertainty

else:
    
    # To run as an script
    from lib import config
    from lib.data import get_dataset, get_mixed_dataset
    from lib.model import get_model
    from lib.analysis import bayesian_predictions, analyse_entropy
    from lib.plot import plot_mixed_uncertainty

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
        `test_mixed.py` is launched as a script, the received
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

# PREDICT FUNCTIONS
# =============================================================================

def predict(model, X_test, y_test, samples=100):
    """Launches the bayesian mixed class predictions
    
    Launches the necessary predictions over `model` to collect the data
    to generate the `mixed classes` table.
    
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
    avg_Ep : ndarray
        List of the averages of the aleatoric uncertainty (Ep) of each
        class. The last position also contains the average of the
        entire image.
    """
    
    # Bayesian stochastic passes
    print(f"\nLaunching {samples} bayesian predictions")
    predictions = bayesian_predictions(model, X_test, samples=samples)
    
    # Analyse entropy
    _, avg_Ep, _ = analyse_entropy(predictions, y_test)
    
    return avg_Ep

# MAIN FUNCTION
# =============================================================================

def test_mixed(epochs):
    """Generates the `mixed classes` table of the `mixed models`
    
    It also generates the `mixed classes` plot of each model.
    
    The table and the plots are saved in the `TEST_DIR` defined in
    `config.py`.
    
    Parameters
    ----------
    epochs : dict
        Dict structure with the epochs of the selected checkpoint for
        testing each model. The keys must correspond to the abbreviated
        name of the dataset of each trained model.
    """
    
    # CONFIGURATION (extracted here as variables just for code clarity)
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
    
    # Bayesian passes
    passes = config.BAYESIAN_PASSES
    
    # Plot parameters
    w = config.PLOT_W
    h = config.PLOT_H
    
    # Table variables
    table = "           First class      Second class    All classes avg.\n"
    table += "         --------------    --------------    --------------\n"
    table += "Image    Ep    Ep mixed    Ep    Ep mixed    EP    EP mixed\n"
    base_str = "{:>5}{:>7.2f}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}\n"
    
    # FOR EVERY DATASET
    # -------------------------------------------------------------------------
    
    for name, dataset in datasets.items():
        
        # Extract dataset classes and features
        num_classes = dataset['num_classes']
        num_features = dataset['num_features']
        class_a = dataset['mixed_class_A']
        class_b = dataset['mixed_class_B']
        
        # Get model dir and mixed model dir
        base_model_dir = (f"{name}_{l1_n}-{l2_n}model_{p_train}train"
                          f"_{learning_rate}lr")
        model_dir = os.path.join(base_model_dir, f"epoch_{epochs[name]}")
        model_dir = os.path.join(base_dir, model_dir)
        mixed_model_dir = base_model_dir + f"_{class_a}-{class_b}mixed"
        mixed_model_dir = os.path.join(mixed_model_dir,
                                       f"epoch_{epochs[name]}")
        mixed_model_dir = os.path.join(base_dir, mixed_model_dir)
        if not os.path.isdir(model_dir) or not os.path.isdir(mixed_model_dir):
            continue
        
        # GET DATA
        # ---------------------------------------------------------------------
        
        # Get dataset
        _, _, X_test, y_test = get_dataset(dataset, d_path, p_train)
        
        # Get mixed dataset
        _, _, m_X_test, m_y_test = get_mixed_dataset(dataset, d_path, p_train,
                                                     class_a, class_b)
        
        # LOAD MODELS
        # ---------------------------------------------------------------------
        
        # Load model
        model = tf.keras.models.load_model(model_dir)
        
        # Load mixed model
        mixed_model = tf.keras.models.load_model(mixed_model_dir)
        
        # LAUNCH PREDICTIONS
        # ---------------------------------------------------------------------
        
        # Mixed tests message
        print(f"\n### Starting {name} mixed test")
        print('#'*80)
        print(f"\nMODEL DIR: {model_dir}")
        print(f"MIXED MODEL DIR: {mixed_model_dir}")
        
        # Launch predictions
        print("\n# Normal prediction")
        avg_Ep = predict(model, X_test, y_test, samples=passes)
        
        # Launch mixed predictions
        print("\n# Mixed classes prediction")
        m_avg_Ep = predict(mixed_model, m_X_test, m_y_test, samples=passes)
        
        # TABLE AND PLOT
        # ---------------------------------------------------------------------
        
        # Save table values
        output_str = base_str.format(name, avg_Ep[class_a], m_avg_Ep[class_a],
                                     avg_Ep[class_b], m_avg_Ep[class_b],
                                     avg_Ep[-1], m_avg_Ep[-1])
        table += output_str
        
        # Plot class uncertainty
        print(f"\n# Generating {name} `class uncertainty` plot")
        data = [[avg_Ep[class_a], avg_Ep[class_b], avg_Ep[-1]],
                [m_avg_Ep[class_a], m_avg_Ep[class_b], m_avg_Ep[-1]]]
        plot_mixed_uncertainty(output_dir, name, epochs[name], data, class_a,
                               class_b, w, h)
    
    # Print table
    print("\n### Mixed tests finished")
    print('#'*80 + "\n")
    print(table, flush=True)
    
    # Save table
    output_file = os.path.join(output_dir, "mixed_classes.txt")
    with open(output_file, 'w') as f:
        print(table, file=f)

if __name__ == "__main__":
    
    # Parse args
    dataset_list = config.DATASETS_LIST
    args = _parse_args(dataset_list)
    
    # Generate parameter structures for main function
    epochs = {}
    for i, name in enumerate(dataset_list):
        epochs[name] = args.epochs[i]
    
    # Launch main function
    test_mixed(epochs)
