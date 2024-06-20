#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from inference_reduced import *

FOREST = [
    LGBMClassifier(random_state=69),
    LGBMClassifier(random_state=69, max_depth=5),
    LGBMClassifier(random_state=69, n_estimators=50),
    LGBMClassifier(random_state=69, n_estimators=150),
    LGBMClassifier(random_state=69, num_leaves=11),
    LGBMClassifier(random_state=69, num_leaves=51),
    LGBMClassifier(random_state=69, min_child_samples=10),
    LGBMClassifier(random_state=69, min_child_samples=30),
    LGBMClassifier(random_state=69, subsample_freq=1, subsample=0.8),
    LGBMClassifier(random_state=69, subsample_freq=1, subsample=0.6),
    LGBMClassifier(random_state=69, subsample_freq=2, subsample=0.8),
    LGBMClassifier(random_state=69, subsample_freq=2, subsample=0.6),
    LGBMClassifier(random_state=69, learning_rate=0.05),
    LGBMClassifier(random_state=69, learning_rate=0.2),
    LGBMClassifier(random_state=69, reg_alpha=0.1),
    LGBMClassifier(random_state=69, reg_lambda=0.1)
]

def forest_predict(trained_forest, X_test, y_test):
    individual_accuracies = []
    sum_probabilities = None
    for clf in trained_forest:
        # Get accuracy of the individual classifier
        accuracy = clf.score(X_test, y_test)
        individual_accuracies.append(accuracy)
        # Get probabilities of each class for the individual classifier
        probabilities = clf.predict_proba(X_test)
        if sum_probabilities is None:
            sum_probabilities = probabilities
        else:
            sum_probabilities += probabilities
    
    # Get the probabilities of the forest as the average of the individual probabilities
    forest_probabilities = sum_probabilities / len(trained_forest)
    # Get the accuracy of the forest
    forest_predictions = np.argmax(forest_probabilities, axis=1)
    test_ok = forest_predictions == y_test
    accuracy_forest = np.mean(test_ok)
    
    return accuracy_forest, individual_accuracies

def main():
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
        top_k_ft_path = next((os.path.join(feature_importances_dir, file) for file in os.listdir(feature_importances_dir) if file.startswith(image_name + "_top_")), None)
        k = int(top_k_ft_path.split("_features.npy")[0].split("_top_")[-1]) # Number of features
        top_k_features = np.load(top_k_ft_path)
        X_train_k = X_train[:, top_k_features]
        X_test_k = X_test[:, top_k_features]

        # Load the trained reduced model
        model = joblib.load("{}/{}_model_{}.joblib".format(models_dir, image_name, k))
        print("\nReduced model with {} features:".format(k))

        # Perform inference with reduced model
        _, _, accuracy_test = lightgbm_predict(model, X_test_k, y_test)
        print("Test Accuracy:   {:.3f}\n".format(accuracy_test))

        # Train forest
        for clf in FOREST:
            clf.fit(X_train_k, y_train)

        # Perform inference with forest
        accuracy_forest, individual_accuracies = forest_predict(FOREST, X_test_k, y_test)
        print("Individual Test Accuracies:")
        for accuracy in individual_accuracies:
            print("{:.3f}".format(accuracy))
        print("Forest Test Accuracy: {:.3f}".format(accuracy_forest))

        # Save forest models
        joblib.dump(FOREST, "{}/forest_models.joblib".format(models_dir))
        
if __name__ == "__main__":
    main()