#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
import numpy as np
import scipy.io
import timeit
import math
import joblib
import json
import sys
import os

# MACROS
# =============================================================================

# Absolute path of the images directory
DATA_PATH = 'data'

# Name of the images information file
IMAGES_FILE_NAME = "BASE_images_15.json"

# Images information file
IMAGES_FILE = os.path.join(DATA_PATH, IMAGES_FILE_NAME)

IMAGES = [("indian_pines_corrected", "indian_pines", 0.15, 16),
          ("KSC", "KSC", 0.15, 13),
          ("paviaU", "paviaU", 0.1, 9),
          ("salinas", "salinas", 0.1, 16)]

ADDRESS = 0

# UTILS

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

# -----------------------------------------------------------------------------

def to_fixed(num, frac_len):
    
    # Multiply num by 2^(number of fractional bits)
    val = num * 2**frac_len
    
    # Round the result to the nearest integer and cast to int
    val = int(round(val))
    
    return val

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

# FUNCTIONS
# =============================================================================

def load_image(image_info):
    """Loads the image and the ground truth from a `mat` file.
    
    If the file is not present in the `data_path` directory, downloads
    the file from the `image_info` url.
    
    Parameters
    ----------
    image_info: dict
        Dict structure with information of the image. Described below.
    
    Returns
    -------
    out: NumPy array, NumPy array
        The image and the ground truth data.
    
    """
    # Image name
    image_name = image_info['key']
    
    # Filenames
    input_file = os.path.join(DATA_PATH, image_info['file'])
    label_file = os.path.join(DATA_PATH, image_info['file_gt'])
    
    # LOAD IMAGE
    try:
        
        # Load image file
        X = scipy.io.loadmat(input_file)[image_name]
    
    except:
        
        # Download image file
        os.system("wget {} -O {}".format(image_info['url'], input_file))
	    
	    # Load image file
        X = scipy.io.loadmat(input_file)[image_name]
    
    # LOAD GROUND TRUTH
    try:
        
        # Load ground truth file
        y = scipy.io.loadmat(label_file)[image_info['key_gt']]
    
    except:
        
        # Download ground truth file
        os.system("wget {} -O {}".format(image_info['url_gt'], label_file))
	    
	    # Load ground truth file
        y = scipy.io.loadmat(label_file)[image_info['key_gt']]
    
    return X, y

def pixel_classification_preprocessing(X, y, image_info):
    """Preprocesses hyperspectral images for pixel classification.
    
    Reshapes the image and the ground truth data, keeps only the labeled
    pixels, normalizes if necesary, and rename the classes to ordered
    integers from 0.
    
    Parameters
    ----------
    X: NumPy array
        The image data.
    y: NumPy array
        The ground truth data.
    image_info: dict
        Dict structure with information of the image. Described below.
    standardization: bool, optional
        Flag to activate data standardization. If `normalization` flag
        is active, only normalization occurs.
    normalization: bool, optional
        Flag to activate data normalization.
    features: int, optional
        Nuber of best features to use. If `0` (default) it uses every
        feature.
    
    Returns
    -------
    out: NumPy array, NumPy array, int, int, int
        The pixels and labels data prepreocessed and the remaining
        number of pixels, features and classes respectively.
    
    """
    # Reshape them to ignore spatiality
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1)
    
    # Keep only labeled pixels
    X = X[y > 0, :]
    y = y[y > 0]
    
    # Rename clases to ordered integers from 0
    for new_class_num, old_class_num in enumerate(np.unique(y)):
        y[y == old_class_num] = new_class_num
    
    # Get image characteristics
    num_pixels, num_features = X.shape
    num_classes = len(np.unique(y))
    
    return X, y, num_pixels, num_features, num_classes

def random_unison(a,b, rstate=None):
  assert len(a) == len(b)
  p = np.random.RandomState(seed=rstate).permutation(len(a))
  return a[p], b[p]

def separate_pixels(X, y, p, seed, image_info, indexes_dir=None):
    """Separate pixels and labels into train, validation and test sets.
    
    Input data has to be preprocessed so classes are consecutively
    named from '0'.
    
    Parameters
    ----------
    X: NumPy array
        The preprocessed pixels.
    y: NumPy array
        The preprocessed labels.
    image_info: dict
        Dict structure with information of the image. Described below.
    indexes_dir: None | String, optional
        If it exists, absolute path of the indexes directory.
    
    Returns
    -------
    out: (NumPy array, NumPy array, NumPy array,
          NumPy array, NumPy array, NumPy array)
        Structures corresponding to:
            (Train pixels, validation pixels, test pixels,
             train labels, validation labels, test labels)
    
    """
    # Get the rest of the data set characteristics
    pixels = image_info['pixels'][1:]
    
    # Generate the data sets sizes
    train_pixels = [int(n*p) for n in pixels]
    train_pixels = [1 if n == 0 else n for n in train_pixels]
    val_pixels = [0 for n in pixels]
    test_pixels = [a - b for a, b in zip(pixels, train_pixels)]
    
    # Calculate sizes of each structure
    num_train_pixels = sum(train_pixels)
    num_val_pixels = sum(val_pixels)
    num_test_pixels = sum(test_pixels)
    
    # Shape of each pixel (some models use complex structures for spaciality)
    pixel_shape = X.shape[1:]
    
    # Prepare structures for train, validation and test data
    X_train = np.zeros((num_train_pixels,) + pixel_shape)
    y_train = np.zeros((num_train_pixels,), dtype=int)
    X_val = np.zeros((num_val_pixels,) + pixel_shape)
    y_val = np.zeros((num_val_pixels,), dtype=int)
    X_test = np.zeros((num_test_pixels,) + pixel_shape)
    y_test = np.zeros((num_test_pixels,), dtype=int)
    
    # Fill train, val and test data structures
    train_end = 0
    val_end = 0
    test_end = 0
    for class_num, (num_train_pixels_class,
                    num_val_pixels_class,
                    num_test_pixels_class) in enumerate(zip(train_pixels,
                                                            val_pixels,
                                                            test_pixels)):
        
        # Get instances of class `class_num`
        class_data = X[y == class_num, :]
        class_labels = y[y == class_num]
        
        # Save train pixels
        train_start = train_end
        train_end = train_start + num_train_pixels_class
        class_start = 0
        class_end = num_train_pixels_class
        X_train[train_start:train_end] = class_data[class_start:class_end]
        y_train[train_start:train_end] = class_labels[class_start:class_end]
        
        # Save val pixels
        val_start = val_end
        val_end = val_start + num_val_pixels_class
        class_start = class_end
        class_end = class_end + num_val_pixels_class
        X_val[val_start:val_end] = class_data[class_start:class_end]
        y_val[val_start:val_end] = class_labels[class_start:class_end]
        
        # Save test pixels
        test_start = test_end
        test_end = test_start + num_test_pixels_class
        class_start = class_end
        class_end = class_end + num_test_pixels_class
        X_test[test_start:test_end] = class_data[class_start:class_end]
        y_test[test_start:test_end] = class_labels[class_start:class_end]
    
    # Shuffle train data
    X_train, y_train = random_unison(X_train, y_train, rstate=seed)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# TREE ANALYSIS FUNCTIONS
# =============================================================================

def tree_max_depth(tree_structure, current_level=0):
    
    if 'split_index' in tree_structure:
        
        # It is a non-leaf node
        current_level += 1
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        return max(tree_max_depth(left_child, current_level),
                   tree_max_depth(right_child, current_level))
        
    else:
        
        # It is a leaf node
        return current_level

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

def tree_num_nodes(tree_structure):
    
    if 'split_index' in tree_structure:
        
        # It is a non-leaf node
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        return (1 + tree_num_nodes(left_child) + tree_num_nodes(right_child))
        
    else:
        
        # It is a leaf node
        return 1

def tree_max_left_child(tree_structure):
    
    if 'split_index' in tree_structure:
        
        # It is a non-leaf node
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        return max(tree_num_nodes(left_child),
                   tree_max_left_child(right_child))
        
    else:
        
        # It is a leaf node
        return 0

# PRINT AND GENERATION FUNCTIONS
# =============================================================================

def generate_lines(tree_structure, lines, current_level=0, level_node=0):
    if 'split_index' in tree_structure:
        
        # It is a non-leaf node
        
        # Propagate into child nodes
        left_child = tree_structure['left_child']
        lines, left_size, left_position = generate_lines(left_child, lines,
                                                         current_level + 1,
                                                         level_node * 2)
        right_child = tree_structure['right_child']
        lines, right_size, right_position = generate_lines(right_child, lines,
                                                           current_level + 1,
                                                           level_node * 2 + 1)
        
        # Get non_leaf node values
        feature = tree_structure['split_feature']
        cmp_value = int(tree_structure['threshold'])
        non_leaf = "{:03d}:{:04d}".format(feature, cmp_value)
        node_size = len(non_leaf)
        
        # Calculate left_padding
        l_spaces = left_position
        l_hyphens = left_size - left_position - 1 - (node_size // 2) # -1 por |
        left_padding = (" " * l_spaces) + "|" + ("-" * l_hyphens)
        
        # Calculate right padding
        r_hyphens = right_position - (node_size // 2) - 1 # -1 por |
        r_spaces = right_size - right_position
        right_padding = ("-" * r_hyphens) + "|" + (" " * r_spaces)
        
        # Generate node, size and position
        node = left_padding + non_leaf + right_padding
        size = len(node)
        position = len(left_padding) + (node_size // 2)
        
        # Introduce the node into the structure
        lines[current_level][level_node] = node
        
        # Return lines structure, size and position
        return lines, size, position
        
    else:
        
        # It is a leaf node
        
        # Get leaf node values
        leaf = " <{:02.3f}> ".format(tree_structure['leaf_value'])
        
        # Generate node, size and position
        node = leaf
        size = len(node)
        position = size // 2
        
        # Introduce the node into the structure
        lines[current_level][level_node] = node
        
        # Generate a column of spaces under the leaf
        next_level_node = level_node
        for next_level in range(current_level + 1, len(lines)):
            next_level_node = next_level_node * 2
            lines[next_level][next_level_node] = " " * size
        
        # Return lines structure, size and position
        return lines, size, position

def print_tree(tree_structure):
    
    levels = tree_max_depth(tree_structure) + 1
    lines = [["" for n in range(2**l)] for l in range(levels)]
    lines, _, _ = generate_lines(tree_structure, lines)
    
    for l, line in enumerate(lines):
        for n, node in enumerate(line):
            print(node, end="")
        print("")

def print_pixel(pixel):
    for feature in pixel:
        print("        Features_din <= \"{:016b}\";".format(int(feature)))
        print("        wait for Clk_period;")

def print_non_leaf(num_feature, cmp_value, right_child_addr):
    
    global ADDRESS
    
    assert num_feature < 2**8, "num_feature exceeded range."
    assert cmp_value < 2**16, "cmp_value exceeded range."
    assert right_child_addr < 2**7, "right_child_addr exceeded range."
    
    non_leaf = "{:08b}{:016b}{:07b}0".format(num_feature,
                                             cmp_value,
                                             right_child_addr)
    
    print("        Addr <= \"{:013b}\";".format(ADDRESS))
    ADDRESS += 1
    print("        Trees_din <= \"{}\";".format(non_leaf))
    print("        wait for Clk_period;")

def print_leaf(pred_value, next_tree, right_corner,
               last_group_tree, last_class_tree, last_tree):
    
    global ADDRESS
    
    # Fixed representation of the prediction value
    leaf = to_fixed_str(pred_value, 16, 13) # pred_value
    
    if not (last_class_tree or last_group_tree):
        # next_tree
        leaf += "{:014b}".format(next_tree)
        # NOT last_group_tree, is_leaf
        leaf += "01"
    else: # last_group_tree
        # Empty 11 bits
        leaf += "00000000000"
        if not right_corner:
            # NOT last_node, NOT last_group/class_node,
            # last_group_tree, is_leaf
            leaf += "00011"
        else: # last_group/class_node (i.e., last_data_received)
            if not last_class_tree:
                # NOT last_node, last_group_node, last_group_tree, is_leaf
                leaf += "00111"
            else: # last_class_node
                if not last_tree:
                    # NOT last_node, last_class_node, last_group_tree, is_leaf
                    leaf += "01111"
                else: # last_node
                    # last_node, last_class_node, last_group_tree, is_leaf
                    leaf += "11111"
    
    print("        Addr <= \"{:013b}\";".format(ADDRESS))
    if last_class_tree and right_corner:
        ADDRESS = 0
    else:
        ADDRESS += 1
    print("        Trees_din <= \"{}\";".format(leaf))
    print("        wait for Clk_period;")

def generate_nodes(tree_structure, next_address, right_corner,
                   last_group_tree, last_class_tree, last_tree):
    
    if 'split_index' in tree_structure:
        
        # It is a non-leaf node
        left_child = tree_structure['left_child']
        right_child = tree_structure['right_child']
        
        # Generate current node
        num_feature = tree_structure['split_feature']
        cmp_value = int(tree_structure['threshold'])
        right_child_addr = tree_num_nodes(left_child) + 1
        print_non_leaf(num_feature, cmp_value, right_child_addr)
        
        # Generate child nodes in pre-order
        generate_nodes(left_child, next_address, False,
                       last_group_tree, last_class_tree, last_tree)
        generate_nodes(right_child, next_address, right_corner,
                       last_group_tree, last_class_tree, last_tree)
        
    else:
        
        # It is a leaf node
        pred_value = tree_structure['leaf_value'] # ¿¿['leaf_weight']??
        next_tree = next_address
        print_leaf(pred_value, next_tree, right_corner,
                   last_group_tree, last_class_tree, last_tree)

def generate_tree_nodes(tree_structure, tree_address, last_group_tree,
                        last_class_tree, last_tree):
    
    next_address = tree_address + tree_num_nodes(tree_structure)
    
    generate_nodes(tree_structure, next_address, True,
                   last_group_tree, last_class_tree, last_tree)
    
    if last_group_tree:
        print("        \n        -- Trees for the next GROUP\n        ")
    
    return next_address

# NODE FORMAT GENERATION FUNCTION
# =============================================================================

def node_format_generation():
    
    # PARAMETROS DE LA ARQUITECTURA
    # =============================
    #bram_bits = 36000
    #node_bits = 32
    #bram_nodes = bram_bits // node_bits
    #class_brams = 8
    #class_nodes = class_brams * bram_nodes
    class_nodes = 2**13
    
    #for image_name, p, num_classes in IMAGES:
    image_name, image_key, p, num_classes = IMAGES[0]
    
    seed = 69
    model_file = "{}_model_{}_{}.joblib".format(image_name, p, seed)
    model = joblib.load(model_file).booster_.dump_model()['tree_info']
    
    # MODELO
    # ======
    # Cantidad de árboles por clase
    trained_class_trees = len(model) // num_classes
    
    # Los N primeros árboles son el árbol 0 de las N clases,
    # los N siguientes el árbol 1, y así sucesivamente.
    # Para extraerlos por clases, reordenamos la estructura.
    ordered_model = [[model[tree_num * num_classes + class_num]
                         for tree_num in range(trained_class_trees)]
                      for class_num in range(num_classes)]
    
    for class_num, class_trees in enumerate(ordered_model):
        
        print("        \n        -- Class {:2d}".format(class_num))
        print("        -----------\n        ")
        
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
        
        g1_end = total_nodes // 3
        g2_end = g1_end * 2
        
        total_nodes = 0
        g2_found = False
        g1_trees = 0
        g2_trees = 0
        for num_tree in range(num_trees):
            tree = class_trees[num_tree]
            total_nodes += tree_num_nodes(tree['tree_structure'])
            if not g2_found and total_nodes > g1_end:
                g1_trees = num_tree
                g2_found = True
            if total_nodes > g2_end:
                g2_trees = num_tree
                break
        
        address = 0
        for num_tree in range(num_trees):
            tree = class_trees[num_tree]
            
            last_group_tree = (num_tree == (g1_trees - 1)
                               or num_tree == (g2_trees - 1))
            last_class_tree = num_tree == (num_trees - 1)
            last_tree = last_class_tree and class_num == (num_classes - 1)
            
            address = generate_tree_nodes(tree['tree_structure'],
                                          address,
                                          last_group_tree,
                                          last_class_tree,
                                          last_tree)

# FEATURES FORMAT GENERATION FUNCTION
# =============================================================================

def features_format_generation():
    
    # Get images information
    with open(IMAGES_FILE, 'r') as f:
        images_information = json.loads(f.read())
    
    # Load image
    image_name, image_key, train_size, num_classes = IMAGES[0]
    image_info = images_information[image_key]
    X, y = load_image(image_info)
    
    # Preprocess image
    (X, y,
     num_pixels,
     num_features,
     num_classes) = pixel_classification_preprocessing(X, y, image_info)
    
    # Separate pixels
    seed = 69
    X_rand, y_rand = random_unison(X, y, rstate=seed)
    (X_train, _, X_test,
     y_train, _, y_test) = separate_pixels(X_rand, y_rand,
                                           train_size, seed,
                                           image_info)
    
    for class_num in range(num_classes):
        
        print("        \n        -- PIXELS OF CLASS {:2d}".format(class_num))
        print("        ---------------------")
        
        for num_pixel, pixel in enumerate(X_test[y_test == class_num]):
            
            print("        \n        -- PIXEL {}\n        ".format(num_pixel))
            
            print_pixel(pixel)

# INFERENCE FUNCTIONS
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

def worst_cycles(trees):
    cycles = 0
    for tree in trees:
        cycles += tree_max_depth(tree['tree_structure']) + 1
    return cycles

def get_worst_cycles(model):
    
    max_class_cycles = 0
    for class_num, class_trees in enumerate(model):
        
        g1_cycles = worst_cycles(class_trees[0])
        g2_cycles = worst_cycles(class_trees[1])
        g3_cycles = worst_cycles(class_trees[2])
        
        if g1_cycles >= g2_cycles and g1_cycles >= g3_cycles:
            class_cycles = g1_cycles * 3
        elif g2_cycles >= g1_cycles and g2_cycles >= g3_cycles:
            class_cycles = g2_cycles * 3 + 1
        elif g3_cycles >= g1_cycles and g3_cycles >= g2_cycles:
            class_cycles = g3_cycles * 3 + 2
        
        if class_cycles > max_class_cycles:
            max_class_cycles = class_cycles
    
    return max_class_cycles

def get_worst_pixel(model, X_test):
    
    worst_pixel = 0
    for pixel in X_test:
        max_class_cycles = 0
        for class_num, class_trees in enumerate(model):
            
            g1_cycles = cycles(class_trees[0], pixel)
            g2_cycles = cycles(class_trees[1], pixel)
            g3_cycles = cycles(class_trees[2], pixel)
            
            if g1_cycles >= g2_cycles and g1_cycles >= g3_cycles:
                class_cycles = g1_cycles * 3
            elif g2_cycles >= g1_cycles and g2_cycles >= g3_cycles:
                class_cycles = g2_cycles * 3 + 1
            elif g3_cycles >= g1_cycles and g3_cycles >= g2_cycles:
                class_cycles = g3_cycles * 3 + 2
            
            if class_cycles > max_class_cycles:
                max_class_cycles = class_cycles
        
        if max_class_cycles > worst_pixel:
            worst_pixel = max_class_cycles
    
    return worst_pixel

def get_pixels(images_information, image_key, train_size, seed=69):
    
    # Load image
    image_info = images_information[image_key]
    X, y = load_image(image_info)
    
    # Preprocess image
    (X, y,
     num_pixels,
     num_features,
     num_classes) = pixel_classification_preprocessing(X, y, image_info)
    
    # Separate pixels
    X_rand, y_rand = random_unison(X, y, rstate=seed)
    (X_train, _, X_test,
     y_train, _, y_test) = separate_pixels(X_rand, y_rand,
                                           train_size, seed,
                                           image_info)
    
    print("{} classes, {} features, {} test pixels.".format(num_classes,
                                                            num_features,
                                                            len(y_test)))
    
    return X_train, y_train, X_test, y_test

def load_model(image_name, train_size, seed=69):
    model_file = "{}_model_{}_{}.joblib".format(image_name, train_size, seed)
    model = joblib.load(model_file)
    model.set_params(device="gpu")
    return model

def get_ordered_model(image_name, train_size, num_classes, seed=69):
    """
    Reorders the model for inference by rearranging the trees in the model.
    The original model is a list of trees, where each tree corresponds to a class.
    The trees are arranged in the following order:
    [c0t0, c0t1, ..., c0tM, c1t0, c1t1, ..., c1tM, ..., cNt0, cNt1, ..., cNtM]
    where N is the number of classes and M is the number of trees per class.
    The reordered model is a list of lists, where each inner list corresponds to a class.
    The trees for each class are grouped together in the inner list:
    [[c0t0, c0t1, ..., c0tM], [c1t0, c1t1, ..., c1tM], ..., [cNt0, cNt1, ..., cNtM]]

    Args:
        image_name (str): The name of the image.
        train_size (int): The size of the training data.
        num_classes (int): The number of classes in the model.
        seed (int, optional): The seed value for random number generation. Defaults to 69.

    Returns:
        list: The ordered model for inference.
    """
    
    # get model
    model_file = "{}_model_{}_{}.joblib".format(image_name, train_size, seed)
    model = joblib.load(model_file).booster_.dump_model()['tree_info']
    
    # reorder model
    trained_class_trees = len(model) // num_classes
    ordered_model = [[model[tree_num * num_classes + class_num]
                         for tree_num in range(trained_class_trees)]
                     for class_num in range(num_classes)]
    
    return ordered_model

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

#def fixed_predict(tree_structure, pixel, total_len=16, frac_len=13):
#    if 'split_index' in tree_structure:
#        
#        # It is a non-leaf node
#        left_child = tree_structure['left_child']
#        right_child = tree_structure['right_child']
#        feature = tree_structure['split_feature']
#        cmp_value = tree_structure['threshold']
#        
#        if pixel[feature] <= cmp_value:
#            return float_predict(left_child, pixel)
#        else:
#            return float_predict(right_child, pixel)
#        
#    else:
#        
#        # It is a leaf node
#        value = tree_structure['leaf_value']
#        MIN_VALUE = -(2**(total_len - 1 - frac_len))
#        MAX_VALUE = 2**(total_len - 1 - frac_len) - 2**(-frac_len)
#        if value < MIN_VALUE:
#            value = MIN_VALUE
#        if value > MAX_VALUE:
#            value = MAX_VALUE
#        return to_fixed(value, frac_len)

def fixed_predict(tree_structure, pixel, total_len=16, frac_len=13):
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
        value = tree_structure['leaf_value']
        value = to_fixed_str(value, total_len, frac_len)
        value = int(value, 2)
        return value

def fixed_accuracy(model, X_test, y_test):
    hits = 0
    for pixel, label in zip(X_test, y_test):
        predictions = [0 for c in range(len(model))]
        for class_num, class_trees in enumerate(model):
            for group in class_trees:
                for tree in group:
                    tree_structure= tree['tree_structure']
                    predictions[class_num] += fixed_predict(tree_structure,
                                                            pixel)
        if np.argmax(predictions) == label:
            hits += 1
    return hits / len(X_test)

def inference():
    
    # Architecture parameters
    class_nodes = 2**13
    
    # FEATURES
    # ========
    
    # Get images information
    with open(IMAGES_FILE, 'r') as f:
        images_information = json.loads(f.read())
    
    # For each image
    for image_name, image_key, train_size, num_classes in IMAGES:
        
        print("\n{}".format(image_name))
        
        # Get test pixels
        _, _, X_test, y_test = get_pixels(images_information,
                                          image_key, train_size)
        
        # Get ordered model
        ordered_model = get_ordered_model(image_name, train_size, num_classes) # TODO IT SHOULD BE "ORGANIZED"
        
        # INFERENCE
        # =========
        
        # For each class
        #     - Keep only the number of trees that fit in the architecture
        #     - Separate them into three groups per class
        final_model = []
        for class_num, class_trees in enumerate(ordered_model):
            
#            total_nodes = 0
#            num_trees = 0
#            for num_tree, tree in enumerate(class_trees):
#                tree_nodes = tree_num_nodes(tree['tree_structure'])
#                if total_nodes + tree_nodes > class_nodes:
#                    num_trees = num_tree
#                    break
#                else:
#                    num_trees += 1
#                    total_nodes += tree_nodes
#            
#            print("{} nodes in {} trees.".format(total_nodes, num_trees))
            num_trees = 200
            
#            # Without sorting
#            selection = [(0, tree)
#                         for tree in class_trees[0:num_trees]]
            
#            # Sort the selected trees by size
#            selection = [(tree['num_leaves'], tree)
#                         for tree in class_trees[0:num_trees]]
            
#            # Sort the selected trees by max depth
#            selection = [(tree_max_depth(tree), tree)
#                         for tree in class_trees[0:num_trees]]
            
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
        (visited_nodes, avg_nodes,
         used_cycles, avg_cycles) = get_cycles(final_model, X_test)
        float_acc = float_accuracy(final_model, X_test, y_test)
        fixed_acc = fixed_accuracy(final_model, X_test, y_test)
        
        print("VISITED_NODES: {} ({} avg.)".format(visited_nodes, avg_nodes))
        print("USED_CYCLES: {} ({} avg.)".format(used_cycles, avg_cycles))
        print("FLOAT_ACC: {}".format(float_acc))
        print("FIXED_ACC: {}".format(fixed_acc))

def lightgbm_predict(trained_model, X_test, y_test,
                     num_iteration=200, num_executions=100):
    """Predicts the test evaluation data of the lightgbm model."""
    
    test_pred = []
    
    # Predict with test data to get time
    start_time = timeit.default_timer()
    print("-- STARTING 100 INFERENCES {:.3f}".format(start_time))
    for i in range(num_executions):
        test_pred.append(trained_model.predict(X_test,
                                               num_iteration=num_iteration,
                                               device='gpu'))
    end_time = timeit.default_timer()
    print("-- FINISHING 100 INFERENCES {:.3f}".format(end_time))
    time = end_time - start_time
    speed = int(X_test.shape[0] * num_executions / time)
    
    # Get accuracy
    num_ok = 0
    total_size = 0
    for test in test_pred:
        test_ok = (test == y_test)
        num_ok += test_ok.sum()
        total_size += test_ok.size
    
    assert total_size != 0, "Empty test."
    test_accuracy = num_ok / total_size
    
    return time, speed, test_accuracy

def inference_lightgbm_measurements():
    
    # FEATURES
    # ========
    
    # Get images information
    with open(IMAGES_FILE, 'r') as f:
        images_information = json.loads(f.read())
    
    # For each image
    for image_name, image_key, train_size, num_classes in IMAGES:
        
        print("\n{}".format(image_name))
        
        # Get test pixels
        _, _, X_test, y_test = get_pixels(images_information,
                                          image_key, train_size)
        
        # Load model
        trained_model = load_model(image_name, train_size)
        
        # INFERENCE
        # =========
        
        (time, speed,
         test_accuracy) = lightgbm_predict(trained_model, X_test, y_test)
        
        print("prediction time: {:.3f}s ({}px/s)".format(time, speed))
        print("test_accuracy:   {:.3f}\n".format(test_accuracy))

def worst_case_cycles():
    
    # Architecture parameters
    class_nodes = 2**13
    
    # Get images information
    with open(IMAGES_FILE, 'r') as f:
        images_information = json.loads(f.read())
    
    # For each image
    for image_name, image_key, train_size, num_classes in IMAGES:
        
        print("\n{}".format(image_name))
        
        # Get test pixels
        _, _, X_test, y_test = get_pixels(images_information,
                                          image_key, train_size)
        
        # Get ordered model
        ordered_model = get_ordered_model(image_name, train_size, num_classes)
        
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
            
#            # Sort the selected trees by size
#            selection = [(tree['num_leaves'], tree)
#                         for tree in class_trees[0:num_trees]]
            
#            # Sort the selected trees by max depth
#            selection = [(tree_max_depth(tree), tree)
#                         for tree in class_trees[0:num_trees]]
            
            # Sort the selected trees by average depth
            selection = [(tree_average_depth(tree), tree)
                         for tree in class_trees[0:num_trees]]
            
            selection.sort()
            
            # Distribute them into the three groups
            class_selected_trees = [[], [], []]
            for i, (_, tree) in enumerate(selection):
                class_selected_trees[i % 3].append(tree)
            
            final_model.append(class_selected_trees)
        
        # Worst case cycles
        #worst_cycles = get_worst_cycles(final_model)
        worst_pixel = get_worst_pixel(final_model, X_test)
        
        #print("WORST_CASE_CYCLES: {}".format(worst_cycles))
        print("WORST_CASE_PIXEL: {}".format(worst_pixel))

# MAIN FUNCTION
# =============================================================================

#inference()
inference_lightgbm_measurements()
#worst_case_cycles()

