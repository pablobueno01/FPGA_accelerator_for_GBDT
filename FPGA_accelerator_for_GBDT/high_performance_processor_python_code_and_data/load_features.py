#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
from inference_reduced import *

LOAD_FEATURES_DIR = "../FPGA_VHDL_code_and_data/load_features"
FEATURE_VALUE_BITS = 16

def main():
    subdir = 'manual'
    # For each image
    for img in IMAGES:
        
        # Get image information
        image_info = IMAGES[img]
        image_name = image_info["key"]
        train_size = image_info["p"]
        num_classes = image_info["num_classes"]
        
        print("\n----------------{}----------------".format(image_name))
        
        # Load image
        X, y = load_image(image_info)
        
        # Preprocess image
        X, y = pixel_classification_preprocessing(X, y)
        
        # Separate data into train and test sets
        _, _, X_test, y_test = separate_pixels(X, y, train_size)
        print("Test pixels: {}".format(X_test.shape[0]))

        # Obtain the reduced data
        top_k_ft_path = next((os.path.join(FEATURE_IMPORTANCES_DIR, subdir, file) for file in os.listdir(FEATURE_IMPORTANCES_DIR+'/'+subdir) if file.startswith(image_name + "_top_")), None)
        k = int(top_k_ft_path.split("_features.npy")[0].split("_top_")[-1]) # Number of features
        top_k_features = np.load(top_k_ft_path)
        X_test_k = X_test[:, top_k_features]

        # Write the features to a file pixel by pixel and class by class
        file_name = "{}/{}_load_features.txt".format(LOAD_FEATURES_DIR, image_name)
        file = open(file_name, "w")
        for class_label in range(num_classes):
            class_pixels = X_test_k[y_test == class_label]
            file.write("\n\t\t-- PIXELS OF CLASS {}\n".format(class_label))
            file.write("\t\t---------------------\n")
            file.write("\t\tclass_label <= std_logic_vector(to_unsigned({}, class_label'length));\n".format(class_label))

            for i, pixel in enumerate(class_pixels):
                file.write("\n\t\t-- PIXEL {}\n".format(i))
                file.write("\t\t-- Load and valid features flags\n")
                file.write("\t\tLoad_features <= '1';\n")
                file.write("\t\tValid_feature <= '1';\n")


                file.write('\n\t\tFeatures_din <= "' + bin(pixel[0])[2:].zfill(FEATURE_VALUE_BITS) + '";\n')
                file.write("\t\twait for Clk_period;\n")
                file.write("\n\t\t-- Reset load flag\n")
                file.write("\t\tLoad_features <= '0';\n\n")
                for j in range(1, k-1):
                    file.write('\t\tFeatures_din <= "' + bin(pixel[j])[2:].zfill(FEATURE_VALUE_BITS) + '";\n')
                    file.write("\t\twait for Clk_period;\n")
                file.write("\n\t\tlast_feature <= '1';\n")
                file.write("\t\tpc_count     <= '1'; -- count pixel\n")
                file.write('\t\tFeatures_din <= "' + bin(pixel[k-1])[2:].zfill(FEATURE_VALUE_BITS) + '";\n')
                file.write("\t\twait for Clk_period;\n")


                file.write("\n\t\t-- Reset count, last and valid flags\n")
                file.write("\t\tpc_count      <= '0';\n")
                file.write("\t\tLast_feature  <= '0';\n")
                file.write("\t\tValid_feature <= '0';\n")
                file.write("\n\t\t-- Wait until inference is complete\n")
                file.write("\t\twait until Finish = '1';\n")
                file.write("\n\t\twait for Clk_period * 1/2;\n")
                file.write("\n\t\tif Dout = class_label then\n")
                file.write("\t\t\thc_count <= '1';\n")
                file.write("\t\tend if;\n")
                file.write("\n\t\twait for Clk_period;\n")
                file.write("\t\thc_count <= '0';\n")
        
        file.write("\n\t\twait;\n")
        file.write("\tend process;\n")
        file.write("end;\n")
        file.close()

        print("Features written to file: {}".format(file_name))

if __name__ == "__main__":
    main()