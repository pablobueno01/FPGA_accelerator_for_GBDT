# DIRECTORIES DESCRIPTION

## data

This directory contains the images dataset and ground truth files.

## models

This directory contains the images trained model files.

## feature_importances

This directory contains the heat maps of the feature importances and the top-k features arrays of each model.

## accuracy_uncertainty_graphics

This directory contains the graphics comparing the accuracy vs the number of features of each model and the graphics from the uncertainty analysis.

## maps

This directory contains the uncertainty maps of the GBDT forests.

## k_means

This directory contains a dictionary for each model with the corresponding centroid for each cmp_value.

## lib

Auxiliar directory for the maps generation.

# FILES DESCRIPTION

## inference_reduced.py

This file contains the python code to execute feature reduction and inference on every image.

Execution: python inference_reduced.py [<th_acc>]

It executes feature reduction with a tolerance of <th_acc> (default 0.01) and saves the trained models in ./models/<th_acc>/. It also saves the feature importance heat maps in ./feature_importances/, the top k features in ./feature_importances/<th_acc>/ and the accuracy vs num. features graphics in ./accuracy_uncertainty_graphics.

## inference_forest.py

This file contains the python code to train the forests with 2, 4, 8 and 16 models.

Execution: python inference_forest.py [<th_acc>]

It uses de top k features in ./feature_importances/<th_acc>/ (default 0.01) and saves the trained forests in ./models/<th_acc>/. If <th_acc> is 0, it uses the top k features in ./feature_importances/manual/ and saves the trained forests in ./models/manual/.

## perform_analysis.py

This file contains the python code to do the uncertainty analysis (without the uncertainty maps).

Execution: python perform_analysis.py [<th_acc>] [<n_models>]

It uses the trained forest with <n_models> (default 16) models in ./models/<th_acc>/ (default 0.025). If <th_acc> is 0, it uses the trained forest in ./models/manual/. It generates the graphics of the uncertainty analysis in ./accuracy_uncertainty_graphics/forest_(<th_acc>|manual)/<n_models>/.

## generate_maps.py

This file contains the python code to generate the uncertainty maps.

Execution: python generate_maps.py [<th_acc>] [<n_models>]

It uses the trained forest with <n_models> (default 16) models in ./models/<th_acc>/ (default 0.025). If <th_acc> is 0, it uses the trained forest in ./models/manual/. It generates the uncertainty maps in ./maps/forest_(<th_acc>|manual)/.

## manual.py

This file contains the python code to train a model <img> with the top <k> features.

Execution: python manual.py <img> <k>

It saves the trained model in ./models/manual/ and the top <k> features in ./feature_importances/manual/.

## ranges.py

This file contains the python code to do a study of the models' nodes.

Execution: python ranges.py

It prints the minimum and maximun values of cmp_value, the maximum value of rel@_right_child and the minimum and maximum values of leaf_value among all 16 models of the forest in ./models/manual/. It also saves in ./k_means/ a dictionary for each image with the corresponding centroid for each cmp_value after performing K-means with 256 centroids.

## inference_fixed.py

This file contains the python code to calculate the accuracy of the models using bit reduction in leaf_value field or centroids in cmp_value field. This file contains the functions to perform specific tests, uncommenting lines or adding new ones.

Execution: python inference_fixed.py

## load_features.py

This file contains the python code to generate the .txt files in ../FPGA_VHDL_code_and_data/load_features/ with the vhdl code for loading the features of the test pixels. It uses de top features of ./feature_importances/manual/.

Execution: load_features.py

## class_rom.py

This file contains the python code to generate the .txt with the vhdl code for the ROMs of each image.

Execution: class_rom.py ['o']

It uses a model from the forest of 16 models in ./models/manual/. By default, it saves the .txt files in ../FPGA_VHDL_code_and_data/class_roms/ (basic accelerator). With the option 'o' (optimal accelerator) , it saves the .txt files in ../FPGA_VHDL_code_and_data_OPTIM/class_roms/ for the ROMs with the trees and in ../FPGA_VHDL_code_and_data_OPTIM/centroids_roms/ for the ROMs with the centroids in ./k_means.