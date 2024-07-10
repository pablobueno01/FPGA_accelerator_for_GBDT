#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
from inference_reduced import *
from scipy.stats import mode
from sklearn.model_selection import ParameterSampler
from lightgbm import LGBMClassifier

FOREST_SIZE = 16

# Define the parameter ranges
param_ranges = {
    'max_depth': range(5, 12),
    'n_estimators': range(50, 300),
    'min_child_samples': range(10, 30),
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'subsample_freq': [1, 2, 3],
    'learning_rate': [0.05, 0.1, 0.2],
    'reg_alpha': [0.0, 0.1, 0.2],
    'reg_lambda': [0.0, 0.1, 0.2],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
}

# Generate random parameter combinations
param_combinations = list(ParameterSampler(param_ranges, n_iter=FOREST_SIZE, random_state=69))
# Set the number of leaves based on the max_depth
for params in param_combinations:
    params['num_leaves'] = 2 ** (params['max_depth'])
    
# Create the list of LGBMClassifier models with random parameters
FOREST = [LGBMClassifier(random_state=69 + i, **params) for i, params in enumerate(param_combinations)]

def forest_predict(trained_forest, X_test, y_test, use_probabilities=True):
    """
    Predicts the class labels or probabilities for a given test dataset using a trained random forest.

    Parameters:
        trained_forest (list): A list of trained decision tree classifiers representing the random forest.
        X_test (array-like): The test dataset features.
        y_test (array-like): The true class labels for the test dataset.
        use_probabilities (bool, optional): If True, the prediction is based on the average of the probabilities 
            of the classifiers. If False, the prediction is based on the mode of the predictions of the classifiers.e.

    Returns:
        accuracy_forest (float): The accuracy of the random forest predictions.
        individual_accuracies (list): A list of accuracies for each individual classifier in the random forest.
    """
    
    individual_accuracies = []
    sum_probabilities = None
    individual_predictions = []

    for clf in trained_forest:
        # Get accuracy of the individual classifier
        accuracy = clf.score(X_test, y_test)
        individual_accuracies.append(accuracy)

        if use_probabilities:
            # Get probabilities of each class for the individual classifier
            probabilities = clf.predict_proba(X_test)
            if sum_probabilities is None:
                sum_probabilities = probabilities
            else:
                sum_probabilities += probabilities
        else:
            # Get predictions of the individual classifier
            individual_predictions.append(clf.predict(X_test))
    
    if use_probabilities:
        # Get the probabilities of the forest as the average of the individual probabilities
        forest_probabilities = sum_probabilities / len(trained_forest)
        # Get the predictions of the forest as the class with the highest probability
        forest_predictions = np.argmax(forest_probabilities, axis=1)
    else:
        # Get the predictions of the forest as the mode of the individual predictions
        forest_predictions, _ = mode(np.array(individual_predictions), axis=0)
        forest_predictions = forest_predictions.ravel()
    
    accuracy_forest = np.mean(forest_predictions == y_test)
    
    return accuracy_forest, individual_accuracies

def main(th_acc=0.01):
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

        # Obtain the reduced data
        top_k_ft_path = next((os.path.join(FEATURE_IMPORTANCES_DIR, '{}'.format(th_acc), file) for file in os.listdir(FEATURE_IMPORTANCES_DIR+'/{}'.format(th_acc)) if file.startswith(image_name + "_top_")), None)
        k = int(top_k_ft_path.split("_features.npy")[0].split("_top_")[-1]) # Number of features
        top_k_features = np.load(top_k_ft_path)
        X_train_k = X_train[:, top_k_features]
        X_test_k = X_test[:, top_k_features]

        # Load the trained reduced model
        model = joblib.load("{}/{}/{}_model_{}.joblib".format(MODELS_DIR, '{}'.format(th_acc), image_name, k))
        print("\nReduced model with {} features:".format(k))

        # Perform inference with reduced model
        _, _, accuracy_test = lightgbm_predict(model, X_test_k, y_test)
        print("Test Accuracy:   {:.3f}\n".format(accuracy_test))

        # Train forest
        print("Training forest with {} models...".format(len(FOREST)))
        for clf in FOREST:
            clf.fit(X_train_k, y_train)

        # Perform inference with forest
        for i in [2, 4, 8, 16]:
            print("\nPerforming inference with forest...")
            accuracy_forest, individual_accuracies = forest_predict(FOREST[:i], X_test_k, y_test)
            print("Forest Test Accuracy ({} models): {:.3f}".format(i, accuracy_forest))

            # Save forest models
            joblib.dump(FOREST[:i], "{}/{}/{}_forest_{}_models.joblib".format(MODELS_DIR, '{}'.format(th_acc), image_name, i))
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        th_acc = float(sys.argv[1])
        if th_acc <= 0:
            print("th_acc must be greater than 0")
            sys.exit(1)
    else:
        th_acc = 0.01

    main(th_acc=th_acc)