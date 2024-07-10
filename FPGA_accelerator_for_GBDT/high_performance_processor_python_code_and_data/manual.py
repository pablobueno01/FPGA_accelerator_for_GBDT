#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from inference_reduced import *
import sys

def main(img, k, load_model=True):
        
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

    # Save feature importance heat map
    importance = get_normalised_feature_importance(model)
    # Sort the features by importance
    most_important_features = np.argsort(importance)[::-1]

    # Perform feature selection
    top_k_features = most_important_features[:k]

    # Calculate the cross-validation accuracy of the new model
    new_model = LGBMClassifier(random_state=69)
    new_model.fit(X_train[:, top_k_features], y_train)
    print("\nReduced model with {} features:".format(k))

    # Perform inference with reduced model
    time, speed, accuracy_test = lightgbm_predict(new_model, X_test[:, top_k_features], y_test)
    print("Test Accuracy:   {:.3f}".format(accuracy_test))

    # Save the reduced model and the top k features
    joblib.dump(new_model, "{}/{}/{}_model_{}.joblib".format(MODELS_DIR, 'manual', image_name, k))
    np.save(os.path.join(FEATURE_IMPORTANCES_DIR, 'manual', "{}_top_{}_features.npy".format(image_name, k)), top_k_features)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python manual.py <img> <k>")
        sys.exit(1)

    img = sys.argv[1]
    k = int(sys.argv[2])

    main(img, k)