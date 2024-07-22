#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from inference_reduced import *
from inference_fixed import *

def tree_num_nodes(tree_structure):
    if 'split_index' in tree_structure:
        # It is a non-leaf node
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        return 1 + tree_num_nodes(left_child) + tree_num_nodes(right_child)
    else:
        # It is a leaf node
        return 1
    
def max_rel_right_child(tree_structure):
    if 'split_index' in tree_structure:
        # It is a non-leaf node
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        left_num_nodes = tree_num_nodes(left_child) + 1
        max_right = max_rel_right_child(right_child)
        return max(left_num_nodes, max_right)
    else:
        # It is a leaf node
        return 0

def min_max_cmp_value(tree_structure):
    if 'split_index' in tree_structure:
        # It is a non-leaf node
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        cmp_value = tree_structure['threshold']

        min_cmp_value = cmp_value
        max_cmp_value = cmp_value
        left_min, left_max = min_max_cmp_value(left_child)
        right_min, right_max = min_max_cmp_value(right_child)

        if left_min < min_cmp_value:
            min_cmp_value = left_min
        if right_min < min_cmp_value:
            min_cmp_value = right_min

        if left_max > max_cmp_value:
            max_cmp_value = left_max
        if right_max > max_cmp_value:
            max_cmp_value = right_max    

        return min_cmp_value, max_cmp_value
    
    else:
        return float('inf'), float('-inf')

def main():

    subdir = 'manual'
    # Architecture parameters
    class_nodes = 2**13

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
        
        # Find the minimum and maximum values of X
        min_val = min(X.flatten())
        max_val = max(X.flatten())
        
        print("Minimum value of X: {}".format(min_val))
        print("Maximum value of X: {}".format(max_val))

        # Obtain the reduced data
        top_k_ft_path = next((os.path.join(FEATURE_IMPORTANCES_DIR, subdir, file) for file in os.listdir(FEATURE_IMPORTANCES_DIR+'/'+subdir) if file.startswith(image_name + "_top_")), None)
        top_k_features = np.load(top_k_ft_path)
        X_k = X[:, top_k_features]

        # Find the minimum and maximum values of X
        min_val = min(X_k.flatten())
        max_val = max(X_k.flatten())
        
        print("Minimum value of X_k: {}".format(min_val))
        print("Maximum value of X_k: {}".format(max_val))

        # Load the trained forest model
        trained_forest = joblib.load("{}/{}/{}_forest_{}_models.joblib".format(MODELS_DIR, subdir,image_name, 16))
        # Get ordered forest
        ordered_forest = get_ordered_forest(trained_forest, num_classes)

        for model_index, ordered_model in enumerate(ordered_forest):
            print("\nModel {}".format(model_index))
            # For each class
            #     - Keep only the number of trees that fit in the architecture
            #     - Separate them into three groups per class
            final_model = []
            for class_num, class_trees in enumerate(ordered_model):
                total_nodes = 0
                num_trees = 0
                for num_tree, tree in enumerate(class_trees):
                   tree_nodes = tree_num_nodes(tree['tree_structure'])
                   if total_nodes + tree_nodes > class_nodes:
                       num_trees = num_tree
                       break
                   else:
                       num_trees += 1
                       total_nodes += tree_nodes
                
                # Sort the selected trees by average depth
                selection = [(tree_average_depth(tree), tree)
                            for tree in class_trees[0:num_trees]]
                selection.sort()
                
                # Distribute them into the three groups
                class_selected_trees = [[], [], []]
                for i, (_, tree) in enumerate(selection):
                    class_selected_trees[i % 3].append(tree)
                
                final_model.append(class_selected_trees)

            # min_threshold = float('inf')
            # max_threshold = float('-inf')
            # for class_num, class_trees in enumerate(final_model):
            #     for group in class_trees:
            #         for tree in group:
            #             tree_structure = tree['tree_structure']
            #             tree_min, tree_max = min_max_cmp_value(tree_structure)
            #             if tree_min < min_threshold:
            #                 min_threshold = tree_min
            #             if tree_max > max_threshold:
            #                 max_threshold = tree_max

            # print("Minimum value of cmp_value: {}".format(min_threshold))
            # print("Maximum value of cmp_value: {}".format(max_threshold))
            max_rel_right = 0
            for class_num, class_trees in enumerate(final_model):
                for group in class_trees:
                    for tree in group:
                        tree_structure = tree['tree_structure']
                        m = max_rel_right_child(tree_structure)
                        if m > max_rel_right:
                            max_rel_right = m

            print("Maximum value of rel_right_child: {}".format(max_rel_right))
if __name__ == "__main__":
    main()