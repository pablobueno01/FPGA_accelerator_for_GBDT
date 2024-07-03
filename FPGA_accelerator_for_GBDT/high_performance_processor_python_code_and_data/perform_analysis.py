#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from inference_reduced import *
from inference_forest import *
import os
import sys
import numpy as np
import math

# UNCERTAINTY FUNCTIONS
#     Global uncertainty (H) corresponds to predictive entropy
#     Aleatoric uncertainty (Ep) corresponds to expected entropy
#     Epistemic uncertainty corresponds to H - Ep subtraction
# =============================================================================

def predictive_entropy(predictions):
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
        result.append(float(group.sum())/len(group))
    
    return result

def accuracy_vs_uncertainty(predictions, y_test, H_limit=2.5, num_groups=25):
    """Generates the `accuracy vs uncertainty` data
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    y_test : ndarray
        Testing data set labels.
    H_limit : float, optional (default: 2.5)
        The max value of the range of uncertainty.
    num_groups : int, optional (default: 25)
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
    test_H = predictive_entropy(predictions)
    
    # Generate a boolean map of hits
    test_ok = np.mean(predictions, axis=0).argmax(axis=1) == y_test
    
    # Uncertainty groups to divide predictions
    H_groups = np.linspace(0.0, H_limit, num_groups + 1)
    
    H_acc = []
    p_pixels = []
    for i in range(num_groups):
        
        # Calculate the average and percentage of pixels of each group
        group = test_ok[(test_H >= H_groups[i]) & (test_H < H_groups[i + 1])]
        if len(group) > 0:
            p_pixels.append(float(len(group))/len(y_test))
            H_acc.append(float(group.sum())/len(group))
    
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
    model_H = predictive_entropy(predictions)
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
    
    It saves the plot in `output_dir` in png format with the name
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
    print("\nSaved {file_name} in {output_dir}".format(file_name=file_name, output_dir=output_dir))

def plot_accuracy_vs_uncertainty(output_dir, acc_data, px_data, w=7, h=4, colours=COLOURS,
                                 H_limit=2.5, num_groups=25):
    """Generates and saves the `accuracy vs uncertainty` plot
    
    It saves the plot in `output_dir` in png format with the name
    `accuracy_vs_uncertainty.png`.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    acc_data : dict
        It contains the `accuracy vs uncertainty` data of each dataset.
        The key must be the dataset name abbreviation.
    px_data : dict
        It contains, for each dataset, the percentage of pixels of each
        uncertainty group. The key must be the dataset name
        abbreviation.
    w : int, optional (default: 7)
        Width of the plot.
    h : int, optional (default: 4)
        Height of the plot.
    colours : dict, optional (default: COLOURS)
        It contains the HEX value of the RGB colour of each dataset.
        The key must be the dataset name abbreviation.
    H_limit : float, optional (default: 2.5)
        The max value of the range of uncertainty for the plot.
    num_groups : int, optional (default: 25)
        Number of groups to divide xticks labels.
    """
    
    # Labels
    H_groups = np.linspace(0.0, H_limit, num_groups + 1)
    labels = ["{:.2f}-{:.2f}".format(H_groups[i], H_groups[i + 1])
              for i in range(num_groups)]
    
    # Xticks
    xticks = np.arange(len(labels))
    
    # Yticks
    yticks = np.arange(0, 1.1, 0.1)
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    for img_name in colours.keys():
        ax.plot(xticks[:len(acc_data[img_name])], acc_data[img_name],
                label="{img_name} acc.".format(img_name=img_name), color=colours[img_name], zorder=3)
        ax.bar(xticks[:len(px_data[img_name])], px_data[img_name],
               label="{img_name} px %".format(img_name=img_name), color=colours[img_name], alpha=0.18,
               zorder=2)
        ax.bar(xticks[:len(px_data[img_name])],
               [-0.007 for i in px_data[img_name]], bottom=px_data[img_name],
               color=colours[img_name], zorder=3)
    
    # Axes label
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Pixels % and accuracy")
    
    # Y axis limit
    ax.set_ylim((0, 1))
    
    # X axis limit
    ax.set_xlim((xticks[0] - 0.5, xticks[-1] + 0.5))
    
    # Y axis minors
    ax.set_yticks(yticks, minor=True)
    
    # Rotate X axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Grid
    ax.grid(axis='y', zorder=1)
    ax.grid(axis='y', which='minor', linestyle='dashed', zorder=1)
    
    # Get legend handles and labels
    lg_handles, lg_labels = plt.gca().get_legend_handles_labels()
    
    # Manual legend to adjust the handles and place labels in a new order
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    ax.add_artist(ax.legend([lg_handles[idx] for idx in order],
                            [lg_labels[idx] for idx in order],
                            loc='upper center', ncol=4,
                            bbox_to_anchor=(0.46, 1.2)))
    
    # Manually added handles upper lines (to match the bars)
    ax.add_artist(ax.legend([lg_handles[0]], [""], framealpha=0,
                            handlelength=1.8, loc='upper center',
                            bbox_to_anchor=(-0.13, 1.146)))
    ax.add_artist(ax.legend([lg_handles[1]], [""], framealpha=0,
                            handlelength=1.8, loc='upper center',
                            bbox_to_anchor=(0.154, 1.146)))
    ax.add_artist(ax.legend([lg_handles[2]], [""], framealpha=0,
                            handlelength=1.8, loc='upper center',
                            bbox_to_anchor=(0.64478, 1.146)))
    ax.add_artist(ax.legend([lg_handles[3]], [""], framealpha=0,
                            handlelength=1.8, loc='upper center',
                            bbox_to_anchor=(0.927, 1.146)))

    # Save
    file_name = "accuracy_vs_uncertainty.png"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("\nSaved {file_name} in {output_dir}".format(file_name=file_name, output_dir=output_dir))

def plot_class_uncertainty(output_dir, name, avg_Ep, avg_H_Ep, w=7, h=4,
                           colours=["#2B4162", "#FA9F42", "#0B6E4F"]):
    """Generates and saves the `class uncertainty` plot of a dataset
    
    It saves the plot in `output_dir` in png format with the name
    `<NAME>_class_uncertainty.png`, where <NAME> is the
    abbreviation of the dataset name.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    name : str
        The abbreviation of the dataset name.
    avg_Ep : ndarray
        List of the averages of the aleatoric uncertainty (Ep) of each
        class. The last position also contains the average of the
        entire image.
    avg_H_Ep : ndarray
        List of the averages of the epistemic uncertainty (H - Ep) of
        each class. The last position also contains the average of the
        entire image.
    w : int, optional
        Width of the plot. Default is 7.
    h : int, optional
        Height of the plot. Default is 4.
    colours : list, optional
        (default: ["#2B4162", "#FA9F42", "#0B6E4F"])
        List with the str format of the HEX value of at least three RGB
        colours.
    """
    
    # Xticks
    xticks = np.arange(len(avg_Ep))
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    ax.bar(xticks, avg_Ep, label="Ep", color=colours[0], zorder=3)
    ax.bar(xticks, avg_H_Ep, bottom=avg_Ep, label="H - Ep",
           color=colours[2], zorder=3)
    
    # Highlight avg border
    ax.bar(xticks[-1], avg_Ep[-1] + avg_H_Ep[-1], zorder=2,
           edgecolor=colours[1], linewidth=4)
    
    # Axes label
    ax.set_xlabel("{} classes".format(name))
    
    # X axis labels
    ax.set_xticks(xticks)
    xlabels = np.append(xticks[:-1], ["AVG"])
    ax.set_xticklabels(xlabels)
    
    # Grid
    ax.grid(axis='y', zorder=1)
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = "{name}_class_uncertainty.png".format(name=name)
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("\nSaved {file_name} in {output_dir}".format(file_name=file_name, output_dir=output_dir))

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

    # Dictionaries to store the data
    reliability_dict = {}
    acc_dict = {}
    px_dict = {}

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

        # Generate class uncertainty plot
        _, class_Ep_avg, class_H_Ep_avg = analyse_entropy(individual_probabilities, y_test)
        plot_class_uncertainty(ACCURACY_GRAPHICS_DIR, image_name, class_Ep_avg, class_H_Ep_avg)

        # Generate reliability diagram data
        reliability_data = reliability_diagram(individual_probabilities, y_test)
        reliability_dict[image_name] = reliability_data

        # Generate accuracy vs uncertainty data
        acc_data, px_data = accuracy_vs_uncertainty(individual_probabilities, y_test)
        acc_dict[image_name] = acc_data
        px_dict[image_name] = px_data
        
    print("\n--------------------------------")

    # Generate reliability diagram plot
    plot_reliability_diagram(ACCURACY_GRAPHICS_DIR, reliability_dict)

    # Generate accuracy vs uncertainty plot
    plot_accuracy_vs_uncertainty(ACCURACY_GRAPHICS_DIR, acc_dict, px_dict)

if __name__ == "__main__":
    main()