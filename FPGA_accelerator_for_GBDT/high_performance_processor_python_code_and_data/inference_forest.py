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

if __name__ == "__main__":
    main()