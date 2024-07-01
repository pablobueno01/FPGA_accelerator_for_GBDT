#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from inference_reduced import *
from inference_forest import *
import os
import sys
import math
import numpy as np

# UNCERTAINTY FUNCTIONS
#     Global uncertainty (H) corresponds to predictive entropy
#     Aleatoric uncertainty (Ep) corresponds to expected entropy
#     Epistemic uncertainty corresponds to H - Ep subtraction
# =============================================================================

def _predictive_entropy(predictions):
    """Calculates the predictive entropy of `predictions`
    
    The predictive entropy corresponds to the global uncertainty (H).
    The correspondent equation can be found in the paper `Bayesian
    Neural Networks to Analyze Hyperspectral Datasets Using Uncertainty
    Metrics`.
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    
    Returns
    -------
    pred_h : ndarray
        Predictive entropy, i.e. global uncertainty, of `predictions`
    """
    
    # Get number of pixels and classes
    _, num_pixels, num_classes = predictions.shape
    
    # Application of the predictive entropy equation
    entropy = np.zeros(num_pixels)
    for p in range(num_pixels):
        for c in range(num_classes):
            avg = np.mean(predictions[..., p, c])
            if avg == 0.0:
                avg = sys.float_info.min
            entropy[p] += avg * math.log(avg)
    
    return -1 * entropy

def _expected_entropy(predictions):
    """Calculates the expected entropy of `predictions`
    
    The expected entropy corresponds to the aleatoric uncertainty (Ep).
    The correspondent equation can be found in the paper `Bayesian
    Neural Networks to Analyze Hyperspectral Datasets Using Uncertainty
    Metrics`.
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    
    Returns
    -------
    pred_ep : ndarray
        Expected entropy, i.e. aleatoric uncertainty, of `predictions`
    """
    
    # Get number of bayesian passes, pixels and classes
    num_tests, num_pixels, num_classes = predictions.shape
    
    # Application of the expected entropy equation
    entropy = np.zeros(num_pixels)
    for p in range(num_pixels):
        for t in range(num_tests):
            class_sum = 0
            for c in range(num_classes):
                val = predictions[t][p][c]
                if val == 0.0:
                    val = sys.float_info.min
                class_sum += val * math.log(val)
            entropy[p] -= class_sum
    
    return entropy/num_tests

# ANALYSIS FUNCTIONS
# =============================================================================

def reliability_diagram(predictions, y_test, num_groups=10):
    """Generates the `reliability diagram` data
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    y_test : ndarray
        Testing data set labels.
    num_groups : int, optional (default: 10)
        Number of groups in which the prediction will be divided
        according to their predicted probability.
    
    Returns
    -------
    result : list of float
        List of the observed probabilities of each one of the predicted
        probability groups.
    """
    
    # Get number of classes
    num_classes = predictions.shape[2]
    
    # Calculate the bayesian samples average
    prediction = np.mean(predictions, axis=0)
    
    # Labels to one-hot encoding
    labels = np.zeros((len(y_test), num_classes))
    labels[np.arange(len(y_test)), y_test] = 1
    
    # Probability groups to divide predictions
    p_groups = np.linspace(0.0, 1.0, num_groups + 1)
    p_groups[-1] += 0.1 # To include the last value
    
    result = []
    for i in range(num_groups):
        
        # Calculate the average of each group
        group = labels[(prediction >= p_groups[i]) &
                       (prediction < p_groups[i + 1])]
        result.append(group.sum()/len(group))
    
    return result

def accuracy_vs_uncertainty(predictions, y_test, H_limit=1.5, num_groups=15):
    """Generates the `accuracy vs uncertainty` data
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    y_test : ndarray
        Testing data set labels.
    H_limit : float, optional (default: 1.5)
        The max value of the range of uncertainty.
    num_groups : int, optional (default: 15)
        Number of groups in which the prediction will be divided
        according to their uncertainty.
    
    Returns
    -------
    H_acc : list of float
        List of the accuracies of each one of the uncertainty groups.
    p_pixels : list of float
        List of the percentage of pixels belonging to each one of the
        uncertainty groups.
    """
    
    # Get predictive entropy
    test_H = _predictive_entropy(predictions)
    
    # Generate a boolean map of hits
    test_ok = np.mean(predictions, axis=0).argmax(axis=1) == y_test
    
    # Uncertainty groups to divide predictions
    H_groups = np.linspace(0.0, H_limit, num_groups + 1)
    
    H_acc = []
    p_pixels = []
    for i in range(num_groups):
        
        # Calculate the average and percentage of pixels of each group
        group = test_ok[(test_H >= H_groups[i]) & (test_H < H_groups[i + 1])]
        p_pixels.append(len(group)/len(y_test))
        H_acc.append(group.sum()/len(group))
    
    return H_acc, p_pixels

def analyse_entropy(predictions, y_test):
    """Calculates the average uncertainty values by class
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    y_test : ndarray
        Testing data set labels.
    
    Returns
    -------
    class_H_avg : ndarray
        List of the averages of the global uncertainty (H) of each
        class. The last position also contains the average of the
        entire image.
    class_Ep_avg : ndarray
        List of the averages of the aleatoric uncertainty (Ep) of each
        class. The last position also contains the average of the
        entire image.
    class_H_Ep_avg : ndarray
        List of the averages of the epistemic uncertainty (H - Ep) of
        each class. The last position also contains the average of the
        entire image.
    """
    
    # Get the uncertainty values
    model_H = _predictive_entropy(predictions)
    model_Ep = _expected_entropy(predictions)
    model_H_Ep = model_H - model_Ep
    
    # Structures for the averages
    num_classes = predictions.shape[2]
    class_H = np.zeros(num_classes + 1)
    class_Ep = np.zeros(num_classes + 1)
    class_H_Ep = np.zeros(num_classes + 1)
    class_px = np.zeros(num_classes + 1, dtype='int')
    
    for px, (H, Ep, H_Ep, label) in enumerate(zip(model_H, model_Ep,
                                                  model_H_Ep, y_test)):
        
        # Label as integer
        label = int(label)
        
        # Accumulate uncertainty values by class
        class_H[label] += H
        class_Ep[label] += Ep
        class_H_Ep[label] += H_Ep
        
        # Count pixels for class average
        class_px[label] += 1
        
        # Accumulate for every class
        class_H[-1] += H
        class_Ep[-1] += Ep
        class_H_Ep[-1] += H_Ep
        
        # Count pixels for global average
        class_px[-1] += 1
    
    # Return averages
    return class_H/class_px, class_Ep/class_px, class_H_Ep/class_px

# PLOT FUNCTIONS
# =============================================================================

# Plots colours
COLOURS = {"indian_pines_corrected": "#FA9F42",
           "KSC": "#0B6E4F",
           "paviaU": "#721817",
           "salinas": "#D496A7"}

def plot_reliability_diagram(output_dir, data, w=7, h=4, colours=COLOURS, num_groups=10):
    """Generates and saves the `reliability diagram` plot
    
    It saves the plot in `output_dir` in pdf format with the name
    `reliability_diagram.png`.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    data : dict
        It contains the `reliability diagram` data of each dataset. The
        key must be the dataset name abbreviation.
    w : int, optional (default: 7)
        Width of the plot.
    h : int, optional (default: 4)
        Height of the plot.
    colours : dict, optional (default: COLOURS)
        It contains the HEX value of the RGB colour of each dataset.
        The key must be the dataset name abbreviation.
    num_groups : int, optional (default: 10)
        Number of groups to divide xticks labels.
    """
    
    # Generate x axis labels and data for the optimal calibration curve
    p_groups = np.linspace(0.0, 1.0, num_groups + 1)
    center = (p_groups[1] - p_groups[0])/2
    optimal = (p_groups + center)[:-1]
    if num_groups <= 10:
        labels = ["{:.1f}-{:.1f}".format(p_groups[i], p_groups[i + 1])
              for i in range(num_groups)]
    else:
        labels = ["{:.2f}-{:.2f}".format(p_groups[i], p_groups[i + 1])
              for i in range(num_groups)]
    
    # Xticks
    xticks = np.arange(len(labels))
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    for img_name in colours.keys():
        ax.plot(xticks[:len(data[img_name])], data[img_name], label=img_name,
                color=colours[img_name])
    ax.plot(xticks, optimal, label="Optimal calibration", color='black',
            linestyle='dashed')
    
    # Axes labels
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    
    # Y axis limit
    ax.set_ylim((0, 1))
    
    # X axis limit
    ax.set_xlim((xticks[0], xticks[-1]))
    
    # Rotate X axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Grid
    ax.grid(axis='y')
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = "reliability_diagram.png"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("\n\nSaved {file_name} in {output_dir}".format(file_name=file_name, output_dir=output_dir))

# PREDICTION FUNCTION
# =============================================================================

def get_forest_individual_probabilities(trained_forest, X_test):
    individual_probabilities = []

    for clf in trained_forest:
        # Get probabilities of each class for the individual classifier
        probabilities = clf.predict_proba(X_test)
        individual_probabilities.append(probabilities)
        
    return np.array(individual_probabilities)

# MAIN FUNCTION
# =============================================================================

def main():

    reliability_dict = {}

    # For each image
    for img in IMAGES:
        
        # Get image information
        image_info = IMAGES[img]
        image_name = image_info["key"]
        train_size = image_info["p"]
        
        print("\n----------------{}----------------".format(image_name))

        # Load image
        X, y = load_image(image_info)
        
        # Preprocess image
        X, y = pixel_classification_preprocessing(X, y)
        
        # Separate data into train and test sets
        _, _, X_test, y_test = separate_pixels(X, y, train_size)
        print("Test pixels: {}".format(X_test.shape[0]))

        # Obtain the reduced data
        top_k_ft_path = next((os.path.join(FEATURE_IMPORTANCES_DIR, file) for file in os.listdir(FEATURE_IMPORTANCES_DIR) if file.startswith(image_name + "_top_")), None)
        k = int(top_k_ft_path.split("_features.npy")[0].split("_top_")[-1]) # Number of features
        top_k_features = np.load(top_k_ft_path)
        X_test_k = X_test[:, top_k_features]

        # -------------FOREST MODEL ANALYSIS-------------
        print('\nForest with {} trees and {} features'.format(FOREST_SIZE, k))

        # Load the trained forest model
        trained_forest = joblib.load("{}/{}_forest_models.joblib".format(MODELS_DIR, image_name))

        # Get the individual probabilities of the forest
        individual_probabilities = get_forest_individual_probabilities(trained_forest, X_test_k)

        # Get the average uncertainty values by class
        # class_H_avg, class_Ep_avg, class_H_Ep_avg = analyse_entropy(individual_probabilities, y_test)
        # print("\nClass H avg:")
        # for class_num, avg in enumerate(class_H_avg):
        #     print("Class {}: {:.3f}".format(class_num, avg))
        
        # print("\nClass Ep avg:")
        # for class_num, avg in enumerate(class_Ep_avg):
        #     print("Class {}: {:.3f}".format(class_num, avg))
        
        # print("\nClass H - Ep avg:")
        # for class_num, avg in enumerate(class_H_Ep_avg):
        #     print("Class {}: {:.3f}".format(class_num, avg))

        # Generate reliability diagram data
        reliability_data = reliability_diagram(individual_probabilities, y_test)
        reliability_dict[image_name] = reliability_data
        
    # Generate reliability diagram plot
    plot_reliability_diagram(ACCURACY_GRAPHICS_DIR, reliability_dict)

if __name__ == "__main__":
    main()