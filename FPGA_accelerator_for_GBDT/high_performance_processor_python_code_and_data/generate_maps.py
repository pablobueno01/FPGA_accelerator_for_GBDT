#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from inference_reduced import *
from perform_analysis import *
from HSI2RGB import HSI2RGB

MAPS_DIR = "maps"

# MAP FUNCTIONS
# =============================================================================

def _uncertainty_to_map(uncertainty, num_classes, slots=0, max_H=0):
    """Groups the uncertainty values received into uncertainty groups
    
    Parameters
    ----------
    uncertainty : ndarray
        Array with the uncertainty values.
    num_classes : int
        Number of classes of the dataset.
    slots : int, optional (default: 0)
        Number of groups to divide uncertainty map values. If set to 0,
        `slots` will be set to `max_H * 10`.
    max_H : float, optional (default: 0)
        Maximum uncertainty value. If set to 0, it will be calculated as
        the maximum value in the `uncertainty` array.
    
    Returns
    -------
    u_map : ndarray
        List with the uncertainty group corresponding to each
        uncertainty value received.
    labels : list of strings
        List of the labels for plotting the `u_map` value groups.
    """
    
    # Actualise `max_H` in case of the default value
    if max_H == 0:
        max_H = np.ceil(np.max(uncertainty) * 10) / 10
    
    # Actualise `slots` in case of the default value
    if slots == 0:
        slots = int(max_H * 10)
    
    # Prepare output structures and ranges
    u_map = np.zeros(uncertainty.shape, dtype="int")
    ranges = np.linspace(0.0, max_H, num=slots+1)
    labels = ["0.0-{:.2f}".format(ranges[1])]
    
    # Populate the output structures
    slot = 1
    start = ranges[1]
    for end in ranges[2:]:
        
        # Fill with the slot number and actualise labels
        u_map[(start <= uncertainty) & (uncertainty <= end)] = slot
        labels.append("{:.2f}-{:.2f}".format(start, end))
        
        # For next iteration
        start = end
        slot +=1
    
    return u_map, labels

def _map_to_img(prediction, shape, colours, metric=None, th=0.0, bg=(0, 0, 0)):
    """Generates an RGB image from `prediction` and `colours`
    
    The prediction itself should represent the index of its
    correspondent colour.
    
    Parameters
    ----------
    prediction : array_like
        Array with the values to represent.
    shape : tuple of ints
        Original shape to reconstruct the image (without channels, just
        height and width).
    colours : list of RGB tuples
        List of colours for the RGB image representation.
    metric : array_like, optional (Default: None)
        Array with the same length of `prediction` to determine a
        metric for plotting or not each `prediction` value according to
        a threshold.
    th : float, optional (Default: 0.0)
        Threshold value to compare with each `metric` value if defined.
    bg : RGB tuple, optional (Default: (0, 0, 0))
        Background colour used for the pixels not represented according
        to `metric`.
    
    Returns
    -------
    img : ndarray
        RGB image representation of `prediction` colouring each group
        according to `colours`.
    """
    
    # Generate RGB image shape
    img_shape = (shape[0], shape[1], 3)
    
    if metric is not None:
        
        # Coloured RGB image that only shows those values where metric
        # is lower to threshold
        return np.reshape([colours[int(p)] if m < th else bg
                           for p, m in zip(prediction, metric)], img_shape)
    else:
        
        # Coloured RGB image of the entire prediction
        return np.reshape([colours[int(p)] for p in prediction], img_shape)
    
# PLOT FUNCTIONS
# =============================================================================
# Maps colours
# 99% accessibility colours (https://sashamaps.net/docs/resources/20-colors/)
MAP_COLOURS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (70, 240, 240), (240, 50, 230), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200),
    (128, 0, 0), (170, 255, 195), (0, 0, 128), (128, 128, 128)]

# MAP_GRADIENTS = [
#     (77, 230, 54), (135, 229, 53), (193, 229, 52), (229, 206, 51),
#     (228, 146, 50), (228, 86, 49), (228, 48, 71), (227, 47, 130),
#     (227, 46, 189), (204, 45, 227), (143, 44, 226), (81, 43, 226),
#     (42, 64, 226), (41, 125, 225), (40, 185, 225), (39, 225, 202)]

# Gradient colours from green juice to ice climber (https://colornamer.robertcooper.me/)
MAP_GRADIENTS = [
    (77, 230, 54), (113, 229, 53), (149, 229, 52), (185, 229, 52), 
    (211, 217, 51), (228, 198, 50), (228, 161, 50), (228, 123, 49), 
    (228, 86, 49), (228, 62, 62), (227, 47, 85), (227, 47, 122), 
    (227, 46, 159), (224, 45, 193), (209, 45, 217), (181, 44, 226), 
    (143, 44, 226), (104, 43, 226), (71, 48, 226), (46, 61, 226), 
    (41, 94, 225), (40, 132, 225), (40, 170, 225), (39, 200, 216), 
    (39, 225, 202)
]

def plot_maps(output_dir, name, shape, num_classes, wl, img, y, pred_map,
              H_map, colours=MAP_COLOURS, gradients=MAP_GRADIENTS, max_H=0, slots=0):
    """Generates and saves the `uncertainty map` plot of a dataset
    
    This plot shows an RGB representation of the hyperspectral image,
    the ground truth, the prediction map and the uncertainty map.
    
    It saves the plot in `output_dir` in png format with the name
    `H_<NAME>.png`, where <NAME> is the abbreviation of the dataset
    name.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    name : str
        The abbreviation of the dataset name.
    shape : tuple of ints
        Original shape to reconstruct the image (without channels, just
        height and width).
    num_classes : int
        Number of classes of the dataset.
    wl : list of floats
        Selected wavelengths of the hyperspectral image for RGB
        representation.
    img : ndarray
        Flattened list of the hyperspectral image pixels normalised.
    y : ndarray
        Flattened ground truth pixels of the hyperspectral image.
    pred_map : ndarray
        Array with the averages of the bayesian predictions.
    H_map : ndarray
        Array with the global uncertainty (H) values.
    colours : list of RGB tuples, optional (default: MAP_COLOURS)
        List of colours for the prediction map classes.
    gradients : list of RGB tuples, optional (default: MAP_GRADIENTS)
        List of colours for the uncertainty map groups of values.
    slots : int, optional (default: 15)
        Number of groups to divide uncertainty map values. If set to 0,
        `slots` will be set to `max_H * 10`.
    max_H : float, optional (default: 0)
        Maximum uncertainty value. If set to 0, it will be calculated as
        the maximum value in the `uncertainty` array.
    """
    
    # PREPARE FIGURE
    # -------------------------------------------------------------------------
    
    # Select shape and size depending on the dataset
    if name=="indian_pines_corrected":
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.set_size_inches(8*shape[1]/96, 8*shape[0]/96)
    elif name=="KSC":
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.set_size_inches(2*shape[1]/96, 2*shape[0]/96)
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        fig.set_size_inches(4*shape[1]/96, shape[0]/96)
    
    # Remove axis
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax4.set_axis_off()
    
    # RGB IMAGE GENERATION
    #     Using HSI2RGB algorithm from paper:
    #         M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O.
    #         Ulfarsson, H. Deborah and J. R. Sveinsson, "Creating RGB
    #         Images from Hyperspectral Images Using a Color Matching
    #         Function," IGARSS 2020 - 2020 IEEE International
    #         Geoscience and Remote Sensing Symposium, 2020,
    #         pp. 2045-2048, doi: 10.1109/IGARSS39084.2020.9323397.
    #     HSI2RGB code from:
    #         https://github.com/JakobSig/HSI2RGB
    # -------------------------------------------------------------------------
    
    # Create and show RGB image (D65 illuminant and 0.002 threshold)
    RGB_img = HSI2RGB(wl, img, shape[0], shape[1], 65, 0.002)
    ax1.imshow(RGB_img)
    ax1.set_title("RGB Image")
    
    # GROUND TRUTH GENERATION
    # -------------------------------------------------------------------------
    
    # Generate and show coloured ground truth
    gt = _map_to_img(y, shape, [(0, 0, 0)] + colours[:num_classes])
    ax2.imshow(gt)
    ax2.set_title("Ground Truth")
    
    # PREDICTION MAP GENERATION
    # -------------------------------------------------------------------------
    
    # Generate and show coloured prediction map
    pred_H_img = _map_to_img(pred_map, shape, colours[:num_classes])
    ax3.imshow(pred_H_img)
    ax3.set_title("Prediction Map")
    
    # UNCERTAINTY MAP GENERATION
    # -------------------------------------------------------------------------
    
    # Create uncertainty map
    u_map, labels = _uncertainty_to_map(H_map, num_classes, slots=slots,
                                        max_H=max_H)
    
    # Generate and show coloured uncertainty map
    if max_H == 0:
        max_H = np.ceil(np.max(u_map) * 10) / 10
    if slots == 0:
        slots = int(max_H * 10)
    H_img = _map_to_img(u_map, shape, gradients[:slots])
    ax4.imshow(H_img)
    ax4.set_title("Uncertainty Map")
    
    # PLOT COMBINED IMAGE
    # -------------------------------------------------------------------------
    
    # Adjust layout between images
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    
    # Save
    file_name = "H_{}.png".format(name)
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("Saved {file_name} in {output_dir}".format(file_name=file_name, output_dir=output_dir))

# MAIN FUNCTION
# =============================================================================

def main(th_acc=0.025, num_models=16):

    if th_acc == 0:
        subdir = 'manual'
    else:
        subdir = '{}'.format(th_acc)

    maps_dir = "{}/{}/{}".format(MAPS_DIR, 'forest_'+subdir, num_models)

    # For each image
    for img in IMAGES:
        
        # Get image information
        image_info = IMAGES[img]
        image_name = image_info["key"]
        
        print("\n----------------{}----------------".format(image_name))

        # Load image
        X, y = load_image(image_info)
        shape = y.shape

        # Preprocess image
        X, y = pixel_classification_preprocessing(X, y, only_labelled=False)

        # Obtain the reduced data
        top_k_ft_path = next((os.path.join(FEATURE_IMPORTANCES_DIR, subdir, file) for file in os.listdir(FEATURE_IMPORTANCES_DIR+'/'+subdir) if file.startswith(image_name + "_top_")), None)
        k = int(top_k_ft_path.split("_features.npy")[0].split("_top_")[-1]) # Number of features
        top_k_features = np.load(top_k_ft_path)
        X_k = X[:, top_k_features]
        # Load the trained forest model
        trained_forest = joblib.load("{}/{}/{}_forest_{}_models.joblib".format(MODELS_DIR, subdir, image_name, num_models))

        # Get the individual probabilities of the forest
        individual_probabilities = get_forest_individual_probabilities(trained_forest, X_k)

        # Generate maps
        X_normalised = X - X.min()
        X_normalised = X_normalised / X_normalised.max()
        pred_map = np.mean(individual_probabilities, axis=0).argmax(axis=1)
        H_map = predictive_entropy(individual_probabilities)
        plot_maps(maps_dir, image_name, shape, image_info["num_classes"], image_info['wl'], X_normalised, y,
                  pred_map, H_map)
        
    print("\n--------------------------------")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        th_acc = float(sys.argv[1])
        num_models = int(sys.argv[2])
    else:
        th_acc = 0.025
        num_models = 16

    main(th_acc=th_acc, num_models=num_models)