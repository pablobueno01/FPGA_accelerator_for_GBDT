#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
from inference_reduced import *

# TREE FUNCTIONS
# =============================================================================
def tree_num_nodes(tree_structure):
    
    if 'split_index' in tree_structure:
        # It is a non-leaf node
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        return (1 + tree_num_nodes(left_child) + tree_num_nodes(right_child))
    else:
        # It is a leaf node
        return 1

def _tree_accum_depth(tree_structure, current_level=0):
    
    if 'split_index' in tree_structure:
        # It is a non-leaf node
        current_level += 1
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        return (_tree_accum_depth(left_child, current_level)
                + _tree_accum_depth(right_child, current_level))
    else:
        # It is a leaf node
        return current_level

def tree_average_depth(tree):
    tree_structure = tree['tree_structure']
    num_leaves = tree['num_leaves']
    return _tree_accum_depth(tree_structure) / num_leaves

# CYCLES FUNCTIONS
# =============================================================================

def tree_cycles(tree_structure, pixel):
    if 'split_index' in tree_structure:
        # It is a non-leaf node
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        feature = tree_structure['split_feature']
        cmp_value = tree_structure['threshold']
        
        if pixel[feature] <= cmp_value:
            #print("LEFT")
            return 1 + tree_cycles(left_child, pixel)
        else:
            #print("RIGHT")
            return 1 + tree_cycles(right_child, pixel)
    else:
        # It is a leaf node
        #print("LEAF")
        return 1
    
def cycles(trees, pixel):
    cycles = 0
    for tree in trees:
        #print_tree(tree['tree_structure'])
        curr_cycles = tree_cycles(tree['tree_structure'], pixel)
        #print(curr_cycles)
        cycles += curr_cycles
    return cycles

def get_cycles(model, X_test):
    
    total_cycles = 0
    total_nodes = 0
    for pixel in X_test:
        max_class_cycles = 0
        max_class_nodes = 0
        for class_num, class_trees in enumerate(model):
            
            g1_cycles = cycles(class_trees[0], pixel)
            g2_cycles = cycles(class_trees[1], pixel)
            g3_cycles = cycles(class_trees[2], pixel)
            
            if (g1_cycles + g2_cycles + g3_cycles) > max_class_nodes:
                max_class_nodes = g1_cycles + g2_cycles + g3_cycles
            
            if g1_cycles >= g2_cycles and g1_cycles >= g3_cycles:
                class_cycles = g1_cycles * 3
            elif g2_cycles >= g1_cycles and g2_cycles >= g3_cycles:
                class_cycles = g2_cycles * 3 + 1
            elif g3_cycles >= g1_cycles and g3_cycles >= g2_cycles:
                class_cycles = g3_cycles * 3 + 2
            
            if class_cycles > max_class_cycles:
                max_class_cycles = class_cycles
        
        total_nodes += max_class_nodes
        total_cycles += max_class_cycles
    
    avg_nodes = total_nodes / len(X_test)
    avg_cycles = total_cycles / len(X_test)
    return total_nodes, avg_nodes, total_cycles, avg_cycles

# FIXED POINT FUNCTIONS
# =============================================================================
# The following functions, `_bits` and `_compliment`, are used to get the
# two's-complement binary representation of positive and negative numbers.
# They have been adapted from:
# 
#     https://michaelwhatcott.com/a-few-bits-of-python/
# 
# -----------------------------------------------------------------------------

_COMPLEMENT = {'1': '0', '0': '1'}

def _compliment(value):
    """Flips each bit of `value` and returns it as str.
    
    Function adapted from https://michaelwhatcott.com/a-few-bits-of-python/
    
    """
    return ''.join(_COMPLEMENT[x] for x in value)

def _bits(number, size_in_bits):
    """Returns str with the two's-complement representation of `number`.
    
    Function adapted from https://michaelwhatcott.com/a-few-bits-of-python/
    
    Parameters
    ----------
    number: int
        Number to be represented in two's-complement.
    size_in_bits: int
        Number of bits of the result.
    
    Returns
    -------
    out: str
        The two's-complement representation of `number` with size
        `size_in_bits`.
    
    Original comments
    ------------------------------------------------------------------------
    The bin() function is *REALLY* unhelpful when working with negative
    numbers.
    It outputs the binary representation of the positive version of that
    number with a '-' at the beginning. Woop-di-do. Here's how to derive
    the two's-complement binary of a negative number:
        
        complement(bin(+n - 1))
    
    `complement` is a function that flips each bit. `+n` is the negative
    number made positive.
    ------------------------------------------------------------------------
    
    """
    if number < 0:
        return _compliment(bin(abs(number) - 1)[2:]).rjust(size_in_bits, '1')
    else:
        return bin(number)[2:].rjust(size_in_bits, '0')
    
def warning_msg(msg, *args, **kwargs):
    """Prints the received arguments as a warning message to stderr.
    
    Parameters
    ----------
    msg: str
        The warning message.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    
    """
    print("[WARNING]: {}".format(msg), *args, file=sys.stderr, **kwargs)

def to_fixed_str(num, total_len, frac_len, verbose=True):
    """Returns str with the binary fixed point representation of a float.
    
    Parameters
    ----------
    num: float
        Value to be represented in fixed point.
    total_len: int
        Number of bits of the fixed point representation.
    frac_len: int
        Number of bits of the fractional part.
    verbose: bool, optional
        When True activates warning messages.
    
    Returns
    -------
    out: str
        Binary fixed point representation of `num` with size `total_len` and
        fractional part of size `frac_len`.
    
    Warnings
    --------
    If `num` is less than the minimum value representable with the received
    `total_len` and `frac_len` values, or if it is greater than the maximum
    value representable, then returns the corresponding truncated value and
    shows a warning message when `verbose` is True.
    
    Raises
    ------
    ValueError:
        If `total_len` is less than `0` or greater than `WORD_LEN`.
        If `frac_len` is less than `0` or greater than `total_len`.
    
    """
    # Calculate minimun and maximum values for the precision range
    MIN_VALUE = -(2**(total_len - 1 - frac_len))
    MAX_VALUE = 2**(total_len - 1 - frac_len) - 2**(-frac_len)
    
    # Test if the value is less than MIN_VALUE
    if num < MIN_VALUE:
        # Warning message
        if verbose:
            wrn_msg = "Value ({}) less than MIN ({}).".format(num, MIN_VALUE)
            warning_msg(wrn_msg)
        
        # Return str with the binary representaton of MIN
        return "1".ljust(total_len, "0")
    
    # Test if the value is greater than MAX_VALUE
    if num > MAX_VALUE:
        # Warning message
        if verbose:
            msg = "Value ({}) greater than MAX ({}).".format(num, MAX_VALUE)
            warning_msg(msg)
        
        # Return str with the binary representaton of MAX
        return "0".ljust(total_len, "1")
    
    # Multiply num by 2^(number of fractional bits)
    val = num * 2**frac_len
    
    # Round the result to the nearest integer and cast to int
    val = int(round(val))
    
    # Return str with the binary representation
    return _bits(val, total_len)

# INFERENCE FUNCTIONS
# =============================================================================
def float_predict(tree_structure, pixel):
    if 'split_index' in tree_structure:
        # It is a non-leaf node
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        feature = tree_structure['split_feature']
        cmp_value = tree_structure['threshold']
        
        if pixel[feature] <= cmp_value:
            return float_predict(left_child, pixel)
        else:
            return float_predict(right_child, pixel)
    else:
        # It is a leaf node
        return tree_structure['leaf_value']

def float_accuracy(model, X_test, y_test):
    hits = 0
    for pixel, label in zip(X_test, y_test):
        predictions = [0 for c in range(len(model))]
        for class_num, class_trees in enumerate(model):
            for group in class_trees:
                for tree in group:
                    tree_structure = tree['tree_structure']
                    predictions[class_num] += float_predict(tree_structure,
                                                            pixel)
        if np.argmax(predictions) == label:
            hits += 1
    return hits / len(X_test)

def fixed_predict(tree_structure, pixel, total_len, frac_len):
    if 'split_index' in tree_structure:
        # It is a non-leaf node
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        feature = tree_structure['split_feature']
        cmp_value = tree_structure['threshold']
        cmp_value = int(cmp_value)
        
        if pixel[feature] <= cmp_value:
            return fixed_predict(left_child, pixel, total_len, frac_len)
        else:
            return fixed_predict(right_child, pixel, total_len, frac_len)
    else:
        # It is a leaf node
        value = tree_structure['leaf_value']
        bits_value = to_fixed_str(value, total_len, frac_len, verbose=False)
        fixed_value = int(bits_value, 2)
        if bits_value[0] == '1':
            fixed_value -= (1 << total_len)
        fixed_value = fixed_value / 2**frac_len
        return fixed_value

def fixed_accuracy(model, X_test, y_test, total_len=16, frac_len=12):
    hits = 0
    for pixel, label in zip(X_test, y_test):
        predictions = [0 for c in range(len(model))]
        for class_num, class_trees in enumerate(model):
            for group in class_trees:
                for tree in group:
                    tree_structure= tree['tree_structure']
                    predictions[class_num] += fixed_predict(tree_structure,
                                                            pixel, total_len, frac_len)
        if np.argmax(predictions) == label:
            hits += 1
    return hits / len(X_test)

# ORDERED FOREST FUNCTIONS
# =============================================================================
def get_ordered_forest(trained_forest, num_classes):
    """
    Reorders each model for inference by rearranging the trees in the model.
    The original model is a list of trees, where each tree corresponds to a class.
    The trees are arranged in the following order:
    [c0t0, c0t1, ..., c0tM, c1t0, c1t1, ..., c1tM, ..., cNt0, cNt1, ..., cNtM]
    where N is the number of classes and M is the number of trees per class.
    The reordered model is a list of lists, where each inner list corresponds to a class.
    The trees for each class are grouped together in the inner list:
    [[c0t0, c0t1, ..., c0tM], [c1t0, c1t1, ..., c1tM], ..., [cNt0, cNt1, ..., cNtM]]

    Args:
        trained_forest (list): The original model as a list of trees.
        num_classes (int): The number of classes in the model.

    Returns:
        list: The ordered models for inference.
    """
    ordered_forest = []
    for model in trained_forest:
        # Get model trees
        model = model.booster_.dump_model()['tree_info']
        # Reorder model
        trained_class_trees = len(model) // num_classes
        ordered_model = [[model[tree_num * num_classes + class_num]
                            for tree_num in range(trained_class_trees)]
                        for class_num in range(num_classes)]
        ordered_forest.append(ordered_model)
    
    return ordered_forest

# MAIN FUNCTION
# =============================================================================

def main(th_acc=0, num_models=16):

    if th_acc == 0:
        subdir = 'manual'
    else:
        subdir = '{}'.format(th_acc)

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
        
        # Separate data into train and test sets
        X_train, y_train, X_test, y_test = separate_pixels(X, y, train_size)
        print("Train pixels: {}\tTest pixels: {}".format(X_train.shape[0], X_test.shape[0]))

        # Obtain the reduced data
        top_k_ft_path = next((os.path.join(FEATURE_IMPORTANCES_DIR, subdir, file) for file in os.listdir(FEATURE_IMPORTANCES_DIR+'/'+subdir) if file.startswith(image_name + "_top_")), None)
        k = int(top_k_ft_path.split("_features.npy")[0].split("_top_")[-1]) # Number of features
        top_k_features = np.load(top_k_ft_path)
        X_test_k = X_test[:, top_k_features]
        X_train_k = X_train[:, top_k_features]

        # Load the trained forest model
        trained_forest = joblib.load("{}/{}/{}_forest_{}_models.joblib".format(MODELS_DIR, subdir,image_name, num_models))
        print('\nForest with {} models and {} features'.format(num_models, k))

        # Get ordered forest
        ordered_forest = get_ordered_forest(trained_forest, num_classes)
        
        final_forest = []
        for model_index, ordered_model in enumerate(ordered_forest):
            print("\nModel {}".format(model_index))
            
            # INFERENCE
            # =========
            
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
            
            # Inference
            #     - Compare floating and fixed point representations
            #     - Calculate the cycles
            print("\nCalculating inference metrics...")
            # (visited_nodes, avg_nodes,
            # used_cycles, avg_cycles) = get_cycles(final_model, X_test_k)
            # float_acc = float_accuracy(final_model, X_test_k, y_test)
            # fixed_acc = fixed_accuracy(final_model, X_test_k, y_test, 7, 3)
            # float_acc = float_accuracy(final_model, X_test_k, y_test)
            # print("FLOAT_ACC: {}".format(float_acc))
            fixed_acc = fixed_accuracy(final_model, X_test_k, y_test, 7, 3)
            print("FIXED_ACC: {}".format(fixed_acc))
            break
            #print("VISITED_NODES: {} ({} avg.)".format(visited_nodes, avg_nodes))
            #print("USED_CYCLES: {} ({} avg.)".format(used_cycles, avg_cycles))
            
            
            

if __name__ == "__main__":
    main(th_acc=0, num_models=16)