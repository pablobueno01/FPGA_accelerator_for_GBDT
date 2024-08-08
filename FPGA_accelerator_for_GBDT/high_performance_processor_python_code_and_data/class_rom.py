#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
from inference_fixed import *
from ranges import tree_num_nodes

CLASS_ROM_DIR = "class_roms"

# Number of bits for each field in a non-leaf tree
FEATURE_BITS = 8
CMP_VALUE_BITS = 16
REL_RIGHT_CHILD_BITS = 7
# Number of bits for each field in a leaf tree
LEAF_VALUE_BITS = 16
LEAF_VALUE_FRAC_BITS = 12
ADDR_NEXT_TREE_BITS = 14

def write_tree(tree_structure, file, addr_next_tree, is_last_tree, rom_addr):
    if 'split_index' in tree_structure:
        # It is a non-leaf node
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        cmp_value = tree_structure['threshold']
        cmp_value = int(cmp_value)
        feature = tree_structure['split_feature']
        # Construct the node
        feature_bin = bin(feature)[2:].zfill(FEATURE_BITS)
        cmp_value_bin = bin(cmp_value)[2:].zfill(CMP_VALUE_BITS)
        rel_right_child_int = tree_num_nodes(left_child) + 1
        rel_right_child_bin = bin(rel_right_child_int)[2:].zfill(REL_RIGHT_CHILD_BITS)
        node_bin = feature_bin + cmp_value_bin + rel_right_child_bin + '0'
        node_int = int(node_bin, 2)
        node_hex = '{:x}'.format(node_int).zfill(8)
        # Write the node to the file
        file.write('\t{} => x"'.format(rom_addr) + node_hex + '",\n')
        # Continue with the children
        write_tree(left_child, file, addr_next_tree, is_last_tree, rom_addr + 1)
        write_tree(right_child, file, addr_next_tree, is_last_tree, rom_addr + rel_right_child_int)
    else:
        # It is a leaf node
        leaf_value = tree_structure['leaf_value']
        # Construct the node
        leaf_value_bin = to_fixed_str(leaf_value, LEAF_VALUE_BITS, LEAF_VALUE_FRAC_BITS)
        node_bin = leaf_value_bin + addr_next_tree + is_last_tree + '1'
        node_int = int(node_bin, 2)
        node_hex = '{:x}'.format(node_int).zfill(8)
        # Write the node to the file
        file.write('\t{} => x"'.format(rom_addr) + node_hex + '",\n')

def main(model_index=0):

    num_models=16
    subdir = "manual"

    for img in IMAGES:
        
        # Get image information
        image_info = IMAGES[img]
        image_name = image_info["key"]
        train_size = image_info["p"]
        num_classes = image_info["num_classes"]
        
        print("\n----------------{}----------------".format(image_name))

        # Obtain the reduced data
        top_k_ft_path = next((os.path.join(FEATURE_IMPORTANCES_DIR, subdir, file) for file in os.listdir(FEATURE_IMPORTANCES_DIR+'/'+subdir) if file.startswith(image_name + "_top_")), None)
        k = int(top_k_ft_path.split("_features.npy")[0].split("_top_")[-1]) # Number of features

        # Load the trained forest model
        trained_forest = joblib.load("{}/{}/{}_forest_{}_models.joblib".format(MODELS_DIR, subdir,image_name, num_models))
        print('\nModel {} of the forest with {} features'.format(model_index, k))

        # Load the trained model
        model = trained_forest[model_index]
        # Get model trees
        model = model.booster_.dump_model()['tree_info']
        # Reorder model
        trained_class_trees = len(model) // num_classes
        ordered_model = [[model[tree_num * num_classes + class_num]
                            for tree_num in range(trained_class_trees)]
                        for class_num in range(num_classes)]
        final_model = get_final_model(ordered_model)

        # ONLY FOR MANUAL TESTING
        # group1=[
        #     {'tree_structure': {'split_index': 0, 'split_feature': 2, 'threshold': 85,
        #                         'left_child': {'split_index': 1, 'split_feature': 6, 'threshold': 42,
        #                                         'left_child': {'leaf_value': 0.3},
        #                                         'right_child': {'leaf_value': 0.5}
        #                                         },
        #                         'right_child': {'leaf_value': 0.7}
        #                         }
        #     },
        #     {'tree_structure': {'split_index': 0, 'split_feature': 15, 'threshold': 34,
        #                         'left_child': {'leaf_value': 0.05},
        #                         'right_child': {'leaf_value': -0.05}
        #                         }
        #     }
        # ]
        # group2=[
        #     {'tree_structure': {'leaf_value': 0.1}}
        # ]
        # class_trees=[group1, group2]
        # final_model=[class_trees]

        # For each class write the trees in a file
        for class_num, class_trees in enumerate(final_model):
            file_name = CLASS_ROM_DIR+'/{}_class_{}.txt'.format(image_name, class_num)
            file = open(file_name, 'w')
            file.truncate(0)
            addr_next_tree = 0  # Address of the next tree
            rom_addr = 0     # Address of the current tree
            for group in class_trees:
                for tree in group:
                    tree_structure = tree['tree_structure']
                    tree_size = tree_num_nodes(tree_structure)
                    addr_next_tree += tree_size
                    is_last_tree = '1' if tree is group[-1] else '0'
                    write_tree(tree_structure, file, bin(addr_next_tree)[2:].zfill(ADDR_NEXT_TREE_BITS), 
                                is_last_tree, rom_addr)
                    rom_addr = addr_next_tree
            print('Class {} written to {}'.format(class_num, file_name))
            file.close()

if __name__ == "__main__":
    main(model_index=0)