#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
import os
import timeit
import scipy.io
import numpy as np
import joblib
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# Images information
IMAGES = {
    "indian_pines": {
        "file": "Indian_pines_corrected.mat",
        "file_gt": "Indian_pines_gt.mat",
        "key": "indian_pines_corrected",
        "key_gt": "indian_pines_gt",
        "url": "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        "url_gt": "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        "p": 0.5,
        "num_classes": 16,
        'num_features': 200,
        'wl' : np.linspace(400, 2500, num=224).take([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
            108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
            134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
            147, 148, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
            174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
            187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
            200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
            213, 214, 215, 216, 217, 218, 220, 221, 223]).tolist(),
        'epochs': 22000
    },
    "KSC": {
        "file": "KSC.mat",
        "file_gt": "KSC_gt.mat",
        "key": "KSC",
        "key_gt": "KSC_gt",
        "url": "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat",
        "url_gt": "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat",
        "p": 0.5,
        "num_classes": 13,
        'num_features': 176,
        'wl' : np.linspace(400, 2500, num=224).take([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
            111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
            137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 167,
            168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
            181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
            194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
            207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 220,
            221, 222]).tolist(),
        'epochs': 41000
    },
    "paviaU": {
        "file": "PaviaU.mat",
        "file_gt": "PaviaU_gt.mat",
        "key": "paviaU",
        "key_gt": "paviaU_gt",
        "url": "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        "url_gt": "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        "p": 0.5,
        "num_classes": 9,
        'num_features': 103,
        'wl' : np.linspace(430, 860, num=115).tolist(),
        'epochs': 1800
    },
    "salinas": {
        "file": "Salinas.mat",
        "file_gt": "Salinas_gt.mat",
        "key": "salinas",
        "key_gt": "salinas_gt",
        "url": "http://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat",
        "url_gt": "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
        "p": 0.5,
        "num_classes": 16,
        'num_features': 204,
        'wl' : np.linspace(400, 2500, num=224).take([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
            108, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
            137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
            150, 151, 152, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
            177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
            190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
            203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
            216, 217, 218, 219, 220, 221, 222]).tolist(),
        'epochs': 4000
    }
}

# Directories to save the results
FEATURE_IMPORTANCES_DIR = "feature_importances"
ACCURACY_GRAPHICS_DIR = "accuracy_uncertainty_graphics"
MODELS_DIR = "models"

# PREPROCESSING FUNCTIONS
# =============================================================================

def load_image(image_info, data_path="./data"):
    """Loads the image and the ground truth from a `mat` file.
    
    If the file is not present in the `data_path` directory, downloads
    the file from the `image_info` url.
    
    Parameters
    ----------
    image_info: dict
        Dict structure with information of the image.
    data_path: str, optional (default "./data")
        Absolute route of the directory of the image files.
    
    Returns
    -------
    X: NumPy array
        The loaded image data.
    y: NumPy array
        The loaded ground truth data.
    
    """
    # Image name
    image_name = image_info['key']
    
    # Filenames
    input_file = os.path.join(data_path, image_info['file'])
    label_file = os.path.join(data_path, image_info['file_gt'])
    
    # LOAD IMAGE
    try:
        
        # Load image file
        X = scipy.io.loadmat(input_file)[image_name]
    
    except:
        
        # Download image file
        os.system("wget {} -O {}".format(image_info['url'], input_file))
        
        # Load image file
        X = scipy.io.loadmat(input_file)[image_name]
    
    # LOAD GROUND TRUTH
    try:
        
        # Load ground truth file
        y = scipy.io.loadmat(label_file)[image_info['key_gt']]
    
    except:
        
        # Download ground truth file
        os.system("wget {} -O {}".format(image_info['url_gt'], label_file))
        
        # Load ground truth file
        y = scipy.io.loadmat(label_file)[image_info['key_gt']]
    
    return X, y

def pixel_classification_preprocessing(X, y, only_labelled=True):
    """Preprocesses hyperspectral images for pixel classification.
    
    Reshapes the image and the ground truth data, keeps only the labeled
    pixels if only_labelled is True, and renames the classes to ordered integers from 0.
    
    Parameters
    ----------
    X: NumPy array
        The image data.
    y: NumPy array
        The ground truth data.
    only_labelled: bool, optional (default True)
        Whether to keep only the labeled pixels or not.
    
    Returns
    -------
    X: NumPy array
        The preprocessed pixels.
    y: NumPy array
        The preprocessed labels.
    
    """
    # Reshape them to ignore spatiality
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1)
    
    # Keep only labeled pixels if only_labelled is True
    if only_labelled:
        X = X[y > 0, :]
        y = y[y > 0]
    
        # Rename classes to ordered integers from 0
        for new_class_num, old_class_num in enumerate(np.unique(y)):
            y[y == old_class_num] = new_class_num
    
    return X, y

def random_index(X, y, seed=69):
  assert len(X) == len(y)
  return np.random.RandomState(seed=seed).permutation(len(X))

def separate_pixels(X, y, p, use_sklearn=True):
    """Separate pixels and labels into train and test sets.
    
    Input data has to be preprocessed so classes are consecutively
    named from '0'.
    
    Parameters
    ----------
    X: NumPy array
        The preprocessed pixels.
    y: NumPy array
        The preprocessed labels.
    p: float
        The percentage of training pixels.
    
    Returns
    -------
    X_train: NumPy array
        Structure corresponding to the train pixels.
    y_train: NumPy array
        Structure corresponding to the train labels.
    X_test: NumPy array
        Structure corresponding to the test pixels.
    y_test: NumPy array, NumPy array, NumPy array)
        Structure corresponding to the test labels.
    
    """

    if use_sklearn:
        from sklearn.model_selection import train_test_split
        (X_train, X_test,
        y_train, y_test) = train_test_split(X, y, test_size=1-p,
                                         random_state=35, stratify=y)
        return X_train, y_train, X_test, y_test
    
    # Shuffle input data
    index = random_index(X, y)
    X = X[index]
    y = y[index]
    
    # Get the number of pixels for each class
    pixels = np.unique(y, return_counts=True)[1]
    
    # Generate the data sets sizes
    train_pixels = [int(n*p) for n in pixels]
    train_pixels = [1 if n == 0 else n for n in train_pixels]
    test_pixels = [a - b for a, b in zip(pixels, train_pixels)]
    
    # Calculate sizes of each structure
    num_train_pixels = sum(train_pixels)
    num_test_pixels = sum(test_pixels)
    
    # Shape of each pixel (some models use complex structures for spaciality)
    pixel_shape = X.shape[1:]
    
    # Prepare structures for train  and test data
    X_train = np.zeros((num_train_pixels,) + pixel_shape)
    y_train = np.zeros((num_train_pixels,), dtype=int)
    X_test = np.zeros((num_test_pixels,) + pixel_shape)
    y_test = np.zeros((num_test_pixels,), dtype=int)
    
    # Fill train and test data structures
    train_end = 0
    test_end = 0
    for class_num, (num_train_pixels_class,
                    num_test_pixels_class) in enumerate(zip(train_pixels,
                                                            test_pixels)):
        
        # Get instances of class `class_num`
        class_data = X[y == class_num, :]
        class_labels = y[y == class_num]
        
        # Save train pixels
        train_start = train_end
        train_end = train_start + num_train_pixels_class
        class_start = 0
        class_end = num_train_pixels_class
        X_train[train_start:train_end] = class_data[class_start:class_end]
        y_train[train_start:train_end] = class_labels[class_start:class_end]
        
        # Save test pixels
        test_start = test_end
        test_end = test_start + num_test_pixels_class
        class_start = class_end
        class_end = class_end + num_test_pixels_class
        X_test[test_start:test_end] = class_data[class_start:class_end]
        y_test[test_start:test_end] = class_labels[class_start:class_end]
    
    # Shuffle train data
    index = random_index(X_train, y_train)
    X_train = X_train[index]
    y_train = y_train[index]
    
    return X_train, y_train, X_test, y_test

# TRAINING FUNCTIONS
# =============================================================================
def obtain_trained_model(X_train, y_train, image_name, load_model=False):
    """
    Obtains a trained model for inference.

    Parameters:
    - X_train (array-like): The input features for training the model.
    - y_train (array-like): The target labels for training the model.
    - image_name (str): The name of the image used for training the model.
    - load_model (bool): Whether to load a pre-trained model or train and save a new one.

    Returns:
    - model: The trained model for inference.
    """

    if not load_model:
        # Train model
        print("Training model...")
        model = LGBMClassifier(importance_type='gain', random_state=69)
        model.fit(X_train, y_train)

        # Save trained model
        joblib.dump(model, "{}/{}_model.joblib".format(MODELS_DIR, image_name))
    else:
        # Load trained model
        model = joblib.load("{}/{}_model.joblib".format(MODELS_DIR, image_name))
    
    return model

# INFERENCE FUNCTIONS
# =============================================================================

def lightgbm_predict(trained_model, X_test, y_test, num_iter=200,
                     power_measurement_iterations=0):
    """Predicts the test evaluation data of the LightGBM model.
    
    Parameters
    ----------
    trained_model:
        The LightGBM trained model.
    X_test: NumPy array
        The test pixels.
    y_test: NumPy array
        The test labels.
    num_iter: int, optional (default 200)
        The `num_iteration` LightGBM parameter that corresponds with the
        number of trees used for each class.
    power_measurement_iterations: int, optional (default 0)
        The number of prediction iterations to perform. This is used to
        increase the time in order to measure power consumption.
    
    Returns
    -------
    time: float
        The computational time of the prediction in seconds.
    speed: int
        The number of pixels predicted per second.
    accuracy: float
        The Top 1 accuracy achieved.
    
    """
    if power_measurement_iterations > 0:
        
        print("--- START ---") # Now you can start measuring power consumption
        
        # Predict with test data and get time
        start_time = timeit.default_timer()
        for i in range(power_measurement_iterations):
            test_pred = trained_model.predict(X_test, num_iteration=num_iter)
        end_time = timeit.default_timer()
        time = end_time - start_time
        speed = int(X_test.shape[0] * power_measurement_iterations / time)
    
    else:
        
        # Predict with test data and get time
        start_time = timeit.default_timer()
        test_pred = trained_model.predict(X_test, num_iteration=num_iter)
        end_time = timeit.default_timer()
        time = end_time - start_time
        speed = int(X_test.shape[0] / time)
    
    # Get accuracy
    test_ok = (test_pred == y_test)
    accuracy = test_ok.sum() / test_ok.size
    
    return time, speed, accuracy

def cross_val_accuracy(model, X_train, y_train, cv=5):
    """
    Calculates the accuracy of a model using cross-validation.

    Parameters:
    - model: The model to evaluate.
    - X_train: The input features for training the model.
    - y_train: The target labels for training the model.
    - cv: The number of cross-validation folds.

    Returns:
    - accuracy: The average accuracy across all folds.
    """
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    accuracy = np.mean(scores)
    return accuracy

# FEATURE IMPORTANCE FUNCTIONS
# =============================================================================

def get_normalised_feature_importance(trained_model):
    """Returns the normalised feature importance with gain criterion.
    
    Parameters
    ----------
    trained_model:
        The trained model.
    
    Returns
    -------
    normalised_importance: NumPy array
        The normalised feature importance.
    
    """
    # Get feature importance with gain criterion
    importance = trained_model.feature_importances_
    
    # Normalize importance to sum up to 1
    normalised_importance = importance / np.sum(importance)
    
    return normalised_importance

def save_feature_importance_heat_map(importance, save_path):
    """
    Saves a heat map of feature importance.

    Parameters:
    importance (numpy.ndarray): Array containing the feature importance values.
    save_path (str): Path to save the heat map image.

    Returns:
    None
    """
    num_features = importance.shape[0]
    
    # Calculate the number of rows needed to fit all features into a grid of 10 columns
    num_rows = (num_features + 9) // 10
    
    # Pad the importance array to make its length a multiple of 10
    importance_padded = np.pad(importance, (0, num_rows * 10 - num_features), 'constant')
    
    # Reshape the padded importance array into a 2D array with 10 columns
    importance_reshaped = importance_padded.reshape(num_rows, 10)
    
    # Save the heat map
    plt.figure(figsize=(12, num_rows))
    sns.heatmap(importance_reshaped, cmap="plasma", annot=True, cbar=True)
    plt.title('Feature Importance Heat Map', fontsize=16)
    plt.xlabel('Feature Index', fontsize=14)
    plt.ylabel('Row Index', fontsize=14)
    plt.savefig(save_path)
    plt.close()

def feature_selection(importance, X_train, y_train, accuracy, image_name, th_acc=0.025):
    """
    Perform feature selection based on feature importance scores.

    Args:
        importance (numpy.ndarray): Array of feature importance scores.
        X_train (numpy.ndarray): Training data features.
        y_train (numpy.ndarray): Training data labels.
        accuracy (float): Target accuracy to achieve.
        image_name (str): Name of the image file to save the plot.
        th_acc (float, optional): Target accuracy tolerance. Defaults to 0.01.

    Returns:
        tuple: A tuple containing the minimum number of features that achieves the target accuracy,
               the indices of the top-k features, and the cross-validation accuracy of the new model.

    """
    # Sort the features by importance
    most_important_features = np.argsort(importance)[::-1]
    # Search for the minimum number of features that achieves the target accuracy
    k = 1
    accuracy_k = 0
    accuracy_values = []
    num_features = []
    while accuracy_k < accuracy - th_acc:
        if k % 10 == 1:
            print("Training model with {} to {} features...".format(k, k+9))
        # Select the top k features
        top_k_features = most_important_features[:k]
        X_train_k = X_train[:, top_k_features]
        # Calculate the cross-validation accuracy of the new model
        new_model = LGBMClassifier(random_state=69)
        accuracy_k = cross_val_accuracy(new_model, X_train_k, y_train)
        accuracy_values.append(accuracy_k)
        num_features.append(k)
        k += 1

    k -= 1  # Decrement k by 1 to get the minimum value
    # Plot accuracy vs number of features
    plt.plot(num_features, accuracy_values, label='Accuracy')
    plt.axhline(y=accuracy, color='r', linestyle='--', label='Target Accuracy: {:.3f}'.format(accuracy))
    plt.fill_between(num_features, accuracy - th_acc, accuracy, color='r', alpha=0.2, label='Target Accuracy Tolerance: {}'.format(th_acc))
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Accuracy vs Number of Features')
    plt.legend()
    plt.savefig('{}/{}_accuracy_vs_num_features_{}.png'.format(ACCURACY_GRAPHICS_DIR, image_name, th_acc))
    plt.close()
    
    return k, top_k_features, accuracy_k

# MAIN FUNCTION
# =============================================================================

def main(load_model=True):
    
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
        X_train, y_train, X_test, y_test = separate_pixels(X, y, train_size)
        print("Train pixels: {}\tTest pixels: {}".format(X_train.shape[0], X_test.shape[0]))
        
        # Obtain trained model
        model = obtain_trained_model(X_train, y_train, image_name, load_model)
        accuracy_cv = cross_val_accuracy(model, X_train, y_train)
        print("\nFull model with {} features:".format(X_train.shape[1]))
        print("Cross-val Accuracy:  {:.3f}".format(accuracy_cv))
        
        # Perform inference
        time, speed, accuracy_test = lightgbm_predict(model, X_test, y_test)
        print("Test Accuracy:   {:.3f}".format(accuracy_test))
        print("Prediction time: {:.3f}s ({}px/s)\n".format(time, speed))

        # Save feature importance heat map
        importance = get_normalised_feature_importance(model)
        save_feature_importance_heat_map(importance, "{}/{}_importance.png".format(FEATURE_IMPORTANCES_DIR, image_name))

        # Perform feature selection
        k, top_k_features, accuracy_k = feature_selection(importance, X_train, y_train, accuracy_cv, image_name)

        # Train reduced model
        new_model = LGBMClassifier(importance_type='gain', random_state=69)
        new_model.fit(X_train[:, top_k_features], y_train)
        print("\nReduced model with {} features:".format(k))
        print("Cross-val Accuracy:  {:.3f}".format(accuracy_k))

        # Perform inference with reduced model
        time, speed, accuracy_test = lightgbm_predict(new_model, X_test[:, top_k_features], y_test)
        print("Test Accuracy:   {:.3f}".format(accuracy_test))
        print("Prediction time: {:.3f}s ({}px/s)\n".format(time, speed))

        # Save the reduced model and the top k features
        joblib.dump(new_model, "{}/{}_model_{}.joblib".format(MODELS_DIR, image_name, k))
        np.save(os.path.join(FEATURE_IMPORTANCES_DIR, "{}_top_{}_features.npy".format(image_name, k)), top_k_features)
        
if __name__ == "__main__":
    main()