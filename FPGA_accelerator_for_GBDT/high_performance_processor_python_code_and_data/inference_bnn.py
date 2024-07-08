#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from inference_reduced import *
from perform_analysis import *
from scipy import io
from sklearn.model_selection import train_test_split

# DATA FUNCTIONS
# =============================================================================

def _load_image(image_info, data_path):
    """Loads the image and the ground truth from a `mat` file
    
    If the file is not present in the `data_path` directory, downloads
    the file from the `image_info` url.
    
    Parameters
    ----------
    image_info : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the hyperspectral images and ground truth files. It can
        be an absolute path or relative from the execution path.
    
    Returns
    -------
    X : ndarray
        Hyperspectral image.
    y : ndarray
        Ground truth.
    """
    
    # Image name
    image_name = image_info['key']
    
    # Generate data path if it does not exist
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    
    # Filenames
    input_file = os.path.join(data_path, image_info['file'])
    label_file = os.path.join(data_path, image_info['file_gt'])
    
    # LOAD IMAGE
    try:
        
        # Load image file
        X = io.loadmat(input_file)[image_name]
    
    except:
        
        # Download image file
        os.system(f"wget {image_info['url']} -O {input_file}")
	    
        # Load image file
        X = io.loadmat(input_file)[image_name]
    
    # LOAD GROUND TRUTH
    try:
        
        # Load ground truth file
        y = io.loadmat(label_file)[image_info['key_gt']]
    
    except:
        
        # Download ground truth file
        os.system(f"wget {image_info['url_gt']} -O {label_file}")
	    
        # Load ground truth file
        y = io.loadmat(label_file)[image_info['key_gt']]
    
    return X, y

def _standardise(X):
    """Standardises a set of hyperspectral pixels
    
    Parameters
    ----------
    X : ndarray
        Set of hyperspectral pixels.
    
    Returns
    -------
    X_standardised : ndarray
        The received set of pixels standardised.
    """
    
    return (X - X.mean(axis=0))/X.std(axis=0)

def _preprocess(X, y, standardisation=False, only_labelled=True):
    """Preprocesses the hyperspectral image and ground truth data
    
    Parameters
    ----------
    X : ndarray
        Hyperspectral image.
    y : ndarray
        Ground truth of `X`.
    standardistion : bool, optional (default: False)
        Flag to activate standardisation.
    only_labelled : bool, optional (default: True)
        Flag to remove unlabelled pixels.
    
    Returns
    -------
    X : ndarray
        Preprocessed data of the hyperspectral image.
    y : ndarray
        Preprocessed data of ground truth.
    """
    
    # Reshape them to ignore spatiality
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1)
    
    if only_labelled:
        
        # Keep only labelled pixels
        X = X[y > 0, :]
        y = y[y > 0]
        
        # Rename clases to ordered integers from 0
        for new_class_num, old_class_num in enumerate(np.unique(y)):
            y[y == old_class_num] = new_class_num
        
        # Regular standardisation
        if standardisation:
            X = _standardise(X)
    
    # Standardise only using labelled pixels for `mean` and `std`
    elif standardisation:
        m = X[y > 0, :].mean(axis=0)
        s = X[y > 0, :].std(axis=0)
        X = (X - m)/s
    
    return X, y

# GET DATASET FUNCTION
# =============================================================================

def get_dataset(dataset, data_path, p_train, seed=35):
    """Returns the preprocessed training and testing data and labels
    
    Parameters
    ----------
    dataset : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    p_train : float
        Represents, from 0.0 to 1.0, the proportion of the dataset that
        will be used for the training set.
    seed : int, optional (default: 35)
        Random seed used to shuffle the data. The same seed will
        produce the same distribution of pixels between train and test
        sets. The default value (35) is just there for reproducibility
        purposes, as it is the used seed in the paper `Bayesian Neural
        Networks to Analyze Hyperspectral Datasets Using Uncertainty
        Metrics`.
    
    Returns
    -------
    X_train : ndarray
        Training data set.
    y_train : ndarray
        Training data set labels.
    X_test : ndarray
        Testing data set.
    y_test : ndarray
        Testing data set labels.
    """
    
    # Load image
    X, y = _load_image(dataset, data_path)
    
    # Preprocess
    X, y = _preprocess(X, y, standardisation=True)
    
    # Separate into train and test data sets
    p_test = 1 - p_train
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=p_test,
                                         random_state=seed, stratify=y)
    
    return X_train, y_train, X_test, y_test

# PREDICT FUNCTIONS
# =============================================================================

def bayesian_predictions(model, X_test, samples=100):
    """Generates bayesian predictions
    
    Parameters
    ----------
    model : TensorFlow Keras Sequential
        Trained bayesian model.
    X_test : ndarray
        Testing data set.
    samples : int, optional (default: 100)
        Number of bayesian passes to perform.
    
    Returns
    -------
    predictions : ndarray
        Array with the bayesian predictions.
    """
    
    # Bayesian stochastic passes
    predictions = []
    for i in range(samples):
        
        # Progress bar
        status = int(78*len(predictions)/samples)
        print('[' + '='*(status) + ' '*(78 - status) + ']', end="\r",
              flush=True)
        
        # Launch prediction
        prediction = model.predict(X_test, verbose=0)
        predictions.append(prediction)
    
    # End of progress bar
    print('[' + '='*78 + ']', flush=True)
    
    return np.array(predictions)

def predict(model, X_test, y_test, samples=100):
    """Launches the bayesian predictions
    
    Launches the necessary predictions over `model` to collect the data
    to generate the `reliability diagram`, the `uncertainty vs accuracy
    plot` and the `class uncertainty` plot of the model.
    
    To generate the `reliability diagram` the predictions are divided
    into groups according to their predicted probability. To generate
    the `uncertainty vs accuracy` plot the predictions are divided into
    groups according to their uncertainty value. For that, it uses the
    default number of groups defined in the `reliability_diagram` and
    the `accuracy_vs_uncertainty` functions of `analysis.py`.
    
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
    rd_data : list of float
        List of the observed probabilities of each one of the predicted
        probability groups.
    acc_data : list of float
        List of the accuracies of each one of the uncertainty groups.
    px_data : list of float
        List of the percentage of pixels belonging to each one of the
        uncertainty groups.
    avg_Ep : ndarray
        List of the averages of the aleatoric uncertainty (Ep) of each
        class. The last position also contains the average of the
        entire image.
    avg_H_Ep : ndarray
        List of the averages of the epistemic uncertainty (H - Ep) of
        each class. The last position also contains the average of the
        entire image.
    """
    
    # Bayesian stochastic passes
    print(f"\nLaunching {samples} bayesian predictions")
    predictions = bayesian_predictions(model, X_test, samples=samples)
    
    # Reliability Diagram
    print("\nGenerating data for the `reliability diagram`", flush=True)
    rd_data = reliability_diagram(predictions, y_test)
    
    # Cross entropy and accuracy
    print("\nGenerating data for the `accuracy vs uncertainty` plot",
          flush=True)
    acc_data, px_data = accuracy_vs_uncertainty(predictions, y_test)
    
    # Analyse entropy
    print("\nGenerating data for the `class uncertainty` plot", flush=True)
    _, avg_Ep, avg_H_Ep = analyse_entropy(predictions, y_test)
    
    return rd_data, acc_data, px_data, avg_Ep, avg_H_Ep

# MAIN FUNCTION
# =============================================================================

def test(epochs):
    """Tests the trained bayesian models
    
    The plots are saved in the `TEST_DIR` defined in `config.py`.
    
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
    d_path = 'data'
    base_dir = os.path.join('bnn_results', 'models')
    datasets = IMAGES
    output_dir = os.path.join('bnn_results', 'accuracy_uncertainty_graphics')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Model parameters
    l1_n = 32
    l2_n = 16
    
    # Training parameters
    p_train = 0.5
    learning_rate = 1.0e-2
    
    # Bayesian passes
    passes = 100
    
    # Plot parameters
    colours = COLOURS
    w = 7
    h = 4
    
    # Plotting variables
    reliability_data = {}
    acc_data = {}
    px_data = {}
    
    # FOR EVERY DATASET
    # -------------------------------------------------------------------------
    
    for name, dataset in datasets.items():
        
        # Get model dir
        model_dir = (f"{name}_{l1_n}-{l2_n}model_{p_train}train"
                     f"_{learning_rate}lr")
        model_dir = os.path.join(model_dir, f"epoch_{epochs[name]}")
        model_dir = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_dir):
            reliability_data[name] = []
            acc_data[name] = []
            px_data[name] = []
            continue
        
        # GET DATA
        # ---------------------------------------------------------------------
        
        # Get dataset
        _, _, X_test, y_test = get_dataset(dataset, d_path, p_train)
        
        # LOAD MODEL
        # ---------------------------------------------------------------------
        
        # Load trained model
        model = tf.keras.models.load_model(model_dir)
        
        # LAUNCH PREDICTIONS
        # ---------------------------------------------------------------------
        
        # Tests message
        print(f"\n### Starting {name} tests")
        print('#'*80)
        print(f"\nMODEL DIR: {model_dir}")
        
        # Launch predictions
        (reliability_data[name],
         acc_data[name],
         px_data[name],
         avg_Ep, avg_H_Ep) = predict(model, X_test, y_test, samples=passes)
        
        # Liberate model
        del model
        
        # IMAGE-RELATED PLOTS
        # ---------------------------------------------------------------------
        
        # Plot class uncertainty
        plot_class_uncertainty(output_dir, name, epochs[name], avg_Ep,
                               avg_H_Ep, w, h)
    
    # End of tests message
    print("\n### Tests finished")
    print('#'*80, flush=True)
        
    # GROUPED PLOTS
    # -------------------------------------------------------------------------
    
    # Plot reliability diagram
    plot_reliability_diagram(output_dir, reliability_data, w, h, colours)
    
    # Plot accuracy vs uncertainty
    plot_accuracy_vs_uncertainty(output_dir, acc_data, px_data, w, h, colours)

if __name__ == "__main__":
    
    # Generate parameter structures for main function
    epochs = {'indian_pines': 22000, 'KSC': 41000, 'paviaU': 1800, 'salinas': 4000}
    
    # Launch main function
    test(epochs)