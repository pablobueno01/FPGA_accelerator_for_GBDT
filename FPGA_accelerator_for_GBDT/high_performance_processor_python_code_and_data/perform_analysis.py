from inference_reduced import *
import os

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
        _, _, X_test, y_test = separate_pixels(X, y, train_size)
        print("Test pixels: {}".format(X_test.shape[0]))

        # Obtain the reduced data
        top_k_ft_path = next((os.path.join(FEATURE_IMPORTANCES_DIR, file) for file in os.listdir(FEATURE_IMPORTANCES_DIR) if file.startswith(image_name + "_top_")), None)
        k = int(top_k_ft_path.split("_features.npy")[0].split("_top_")[-1]) # Number of features
        top_k_features = np.load(top_k_ft_path)
        X_test_k = X_test[:, top_k_features]

        # Load the trained reduced model
        model = joblib.load("{}/{}_model_{}.joblib".format(MODELS_DIR, image_name, k))
        print("\nReduced model with {} features:".format(k))

        # Obtain predictions
        prediction = model.predict_proba(X_test_k)
        print("Prediction dimensions: {}".format(prediction.shape))

if __name__ == "__main__":
    main()