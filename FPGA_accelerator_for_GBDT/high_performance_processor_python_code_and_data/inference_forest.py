#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from inference_reduced import *

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
        accuracy_cv = cross_val_accuracy(model, X_train_k, y_train)
        print("\nReduced model with {} features:".format(k))
        print("Cross-val Accuracy:  {:.3f}".format(accuracy_cv))

        # Perform inference with reduced model
        time, speed, accuracy_test = lightgbm_predict(model, X_test_k, y_test)
        print("Test Accuracy:   {:.3f}".format(accuracy_test))
        print("Prediction time: {:.3f}s ({}px/s)\n".format(time, speed))

if __name__ == "__main__":
    main()