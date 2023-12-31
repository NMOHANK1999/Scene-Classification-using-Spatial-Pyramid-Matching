import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
        
    hist,_ = np.histogram(wordmap.flatten(), bins = range(K+1))
    a = np.sum(hist) 
    hist = hist / a
    return hist
        
    

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    
    histogram_total =[]
    for layer in range(L, -1, -1):
        
        no_of_layers = 2 ** layer
        
        sub_wmaps = []
        for sub_y_wmaps in np.array_split(wordmap, no_of_layers, axis = 0):
            for sub_xy_wmaps in np.array_split(sub_y_wmaps, no_of_layers, axis = 1):
                sub_wmaps.append(sub_xy_wmaps)
                
                
        if layer <= 1:
            weight = 2 ** (-L)
        else:
            weight = 2 ** (layer - L -1)
        
        histogram_all = [] 
        for i in range(no_of_layers * no_of_layers):
            histogram =  get_feature_from_wordmap(opts, sub_wmaps[i])
            weighted_histogram = histogram * weight
            histogram_all = np.hstack([histogram_all, weighted_histogram])#this may be the reason of error
            # if i==0:
            #     histogram_total = weighted_histogram
            # else:
            #     histogram_total += weighted_histogram
            
    

    #return histogram_all/np.sum(histogram_all)
    return histogram_all
    
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    
    1_loads an image,
    2_extract word map from the image, 
    3_computes the SPM, 
    4_returns the computed feature
    '''
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    
    img = Image.open(join(data_dir, img_path))
    img = np.array(img).astype(np.float32)/255
    
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    
    return feature
    
    



def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    features = []

    # ----- TODO -----
    features = []
    for training in train_files:
        feat = get_image_feature(opts, training, dictionary)
        if features == []:
            features = feat.reshape(1,-1) 
        else:
            features = np.row_stack([features, feat])


    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''
    min_val = np.minimum(word_hist, histograms)
    hist_inter_sim = np.sum(min_val, axis = 1)
    return (hist_inter_sim)
    
   
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    
    
    ############ From here onwards
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    features = trained_system['features']
    test_groups = 8
    conf_mat = np.zeros((test_groups,test_groups))
    
    for i in range(len(test_files)):
        img_path = test_files[i]
        #print(img_path)
        img = Image.open(join(data_dir, img_path))
        img = np.array(img).astype(np.float32)/255
        true_label = test_labels[i]
        wordmap_img = visual_words.get_visual_words(opts, img, dictionary)
        feat_img = get_feature_from_wordmap_SPM(opts, wordmap_img)
        distance = distance_to_set(feat_img, features)
        pred_label = train_labels[np.argmax(distance)]
        conf_mat[true_label, pred_label] += 1 
    
    
    acc = np.trace(conf_mat)/np.sum(conf_mat)
    print(acc)
    
    return conf_mat, acc
    