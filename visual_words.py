import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
from sklearn.cluster import KMeans

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # ADDING CODE HERE
        
    if(len(np.array(img).shape) ==3):
           
        img = skimage.color.rgb2lab(img) 
        
        [B,G,R] = np.dsplit(img, img.shape[-1])
        B = B[:,:,0]
        G = G[:,:,0]
        R = R[:,:,0]
    
    else:
        R = np.array(img.copy())
        G = np.array(img.copy())
        B = np.array(img.copy())
        
        
    B_norm = np.abs(B)/np.max(B)
    G_norm = np.abs(G)/np.max(G)
    R_norm = np.abs(R)/np.max(R)
    
    Norm_img = np.dstack([B_norm, G_norm, R_norm])
    
    response_filter = np.zeros((Norm_img.shape[0], Norm_img.shape[1], len(filter_scales) * 4 * Norm_img.shape[2]))
    #response_filter = []
    c= 0
    for scale in (filter_scales):
        for channel in range(Norm_img.shape[2]):
            
            filter_1_img = scipy.ndimage.gaussian_filter(Norm_img[:, :, channel], sigma = scale, order=0, mode='constant', cval=0.0)
            filter_2_img = scipy.ndimage.gaussian_laplace(Norm_img[:, :, channel], sigma = scale, mode='constant', cval=0.0)
            filter_3_img = scipy.ndimage.gaussian_filter(Norm_img[:, :, channel], sigma = scale, order=[1,0], mode='nearest', cval=0.0) # mode nearest
            filter_4_img = scipy.ndimage.gaussian_filter(Norm_img[:, :, channel], sigma = scale, order=[1,1], mode='nearest', cval=0.0)
        
            response_filter[:, :, c * 4] = filter_1_img
            response_filter[:, :, c * 4 + 1] = filter_2_img
            response_filter[:, :, c * 4 + 2] = filter_3_img
            response_filter[:, :, c * 4 + 3] = filter_4_img
            c += 1
    
    return response_filter
    




def compute_dictionary_one_image(opts, train_files):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    data_dir = opts.data_dir
    alpha = opts.alpha
        
    filtered_training_data = []
    
    for files in (train_files):
        
        train_img_path = join(data_dir, files)
        img = Image.open(train_img_path)
        img = np.array(img).astype(np.float32)/255
        
        filter_responses = extract_filter_responses(opts, img)
         
        #add alpha random variables
        x_rand = np.random.randint(0, img.shape[0], alpha)
        y_rand = np.random.randint(0, img.shape[1], alpha)
        reduced_filter_response = filter_responses[x_rand, y_rand, :]
        if not np.any(filtered_training_data):
            filtered_training_data = reduced_filter_response
        else:
            filtered_training_data = np.vstack([filtered_training_data, reduced_filter_response])
            
    return filtered_training_data


def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''    
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    #read about multiprocessing, pool and pool.map. #mapper of the map reduce paradigm = sublinear prcess, multiprocessing right here. 
    filtered_training_data = compute_dictionary_one_image(opts, train_files)
        
    kmeans = KMeans(n_clusters=K).fit(filtered_training_data)
    dictionary = kmeans.cluster_centers_
    
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    height = img.shape[0]
    length  =img.shape[1]
    
    filter_responses = extract_filter_responses(opts, img)
    F3 = filter_responses.shape[2]
    filter_responses_2D = filter_responses.reshape(height*length, F3)
    distances = scipy.spatial.distance.cdist(filter_responses_2D, dictionary)
    wordmap = np.argmin(distances, axis = 1)
    
    return( wordmap.reshape(height, length))
    
    

