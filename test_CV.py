# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 18:46:40 2023

@author: NIshanth Mohankumar
"""

from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage
import scipy
import math 
from sklearn.cluster import KMeans

import util
import visual_words
import visual_recog
from opts import get_opts

import os

# Get the current working directory
current_directory = os.getcwd()

opts = get_opts()

img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
img = Image.open(img_path)


img = np.array(img).astype(np.float32)/255


# a = len(np.array(img).shape)
# a  = len(img.shape)
# a = img.mode
# img.show()

# np.max(R)
# a = R.dtype

filter_scales = opts.filter_scales
    # ----- TODO -----
    #pass
    
if(len(np.array(img).shape) ==3):
           
    img = skimage.color.rgb2lab(img) 
            
    [B,G,R] = np.dsplit(img, img.shape[-1])
    B = B[:,:,0]
    G = G[:,:,0]
    R = R[:,:,0]
        
elif (len(np.array(img).shape) == 1):
    R = np.array(img.copy())
    G = np.array(img.copy())
    B = np.array(img.copy())
            
            
        
B_norm = np.abs(B)/np.max(B)
G_norm = np.abs(G)/np.max(G)
R_norm = np.abs(R)/np.max(R)
        
        
# filter_B_img = scipy.ndimage.gaussian_filter(B_norm, sigma = filter_scales[1], order=0, mode='constant', cval=0.0)
# filter_B4_img = scipy.ndimage.gaussian_filter1d(B_norm, sigma = filter_scales[1], axis = 0, order=1, mode='constant', cval=0.0)
        
# plt.imshow(B_norm)
        
# plt.imshow(filter_B_img)
        
# plt.imshow(filter_B4_img)
  



Norm_img = np.dstack([B_norm, G_norm, R_norm])
#filters = []

response_filter = np.zeros((Norm_img.shape[0], Norm_img.shape[1], len(filter_scales) * 4 * Norm_img.shape[2]))
#response_filter = []
c= 0
for a, scale in enumerate(filter_scales):
    for channel in range(Norm_img.shape[2]):
        
        filter_1_img = scipy.ndimage.gaussian_filter(Norm_img[:, :, channel], sigma = scale, order=0, mode='constant', cval=0.0)
        filter_2_img = scipy.ndimage.gaussian_laplace(Norm_img[:, :, channel], sigma = scale, mode='constant', cval=0.0)
        filter_3_img = scipy.ndimage.gaussian_filter1d(Norm_img[:, :, channel], sigma = scale, axis = 0, order=1, mode='constant', cval=0.0)
        filter_4_img = scipy.ndimage.gaussian_filter1d(Norm_img[:, :, channel], sigma = scale, axis = 1, order=1, mode='constant', cval=0.0)
        
        

        #response_filter = np.dstack([filter_1_img, filter_2_img, filter_3_img, filter_4_img]) 

        # response_filter.append(filter_1_img)
        # response_filter.append(filter_2_img)
        # response_filter.append(filter_3_img)
        # response_filter.append(filter_4_img)
        

        response_filter[:, :, c * 4] = filter_1_img
        response_filter[:, :, c * 4 + 1] = filter_2_img
        response_filter[:, :, c * 4 + 2] = filter_3_img
        response_filter[:, :, c * 4 + 3] = filter_4_img
        c += 1

    

n_scale = len(opts.filter_scales)
plt.figure(1)

for i in range(n_scale*4): #is there an error here?? arent there supposed to be 24 images per thing
    plt.subplot(n_scale, 4, i+1)
    resp = response_filter[:, :, i*3 : i*3 + 3]
    resp_min = resp.min(axis=(0,1), keepdims=True)
    resp_max = resp.max(axis=(0,1), keepdims=True)
    resp = (resp - resp_min)/(resp_max - resp_min)
    plt.imshow(resp)
    plt.axis("off")

plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,wspace=0.05,hspace=0.05)
plt.show()
    
##############################################def compute_dictionary(opts, n_worker=1):
data_dir = opts.data_dir
feat_dir = opts.feat_dir
out_dir = opts.out_dir
K = opts.K
alpha = opts.alpha


train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()

filtered_training_data = []

for i, files in enumerate(train_files):
    
    #if stop < 10:
    train_img_path = join(opts.data_dir, files)
    img = Image.open(train_img_path)
    img = np.array(img).astype(np.float32)/255
    
    filter_responses = visual_words.extract_filter_responses(opts, img)
     
    #add alpha random variables
    x_rand = np.random.randint(0, img.shape[0], alpha)
    y_rand = np.random.randint(0, img.shape[1], alpha)
    reduced_filter_response = filter_responses[x_rand, y_rand, :]
    if not np.any(filtered_training_data):
        filtered_training_data = reduced_filter_response
    else:
        filtered_training_data = np.vstack([filtered_training_data, reduced_filter_response])
    #else:
    #    break
    #stop += 1
    
kmeans = KMeans(n_clusters=K).fit(filtered_training_data)
dictionary = kmeans.cluster_centers_

#this is already in the code
np.save(join(out_dir, 'dictionary.npy'), dictionary)

##################1.3 :Computing Visual Words
img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
img = Image.open(img_path)
img = np.array(img).astype(np.float32)/255
dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))

#wordmap = visual_words.get_visual_words(opts, img, dictionary)
#util.visualize_wordmap(wordmap)

####### above section already in main, following, get_visual_words(opts, img, dictionary): 
filter_responses = visual_words.extract_filter_responses(opts, img)
F3 = filter_responses.shape[2]

filter_responses_2D = filter_responses.reshape(-1, filter_responses.shape[2])

distances = scipy.spatial.distance.cdist(filter_responses_2D, dictionary)

wordmap = np.argmin(distances, axis = 1)

wordmap = wordmap.reshape(filter_responses.shape[0], filter_responses.shape[1])

plt.imshow(wordmap)



#############get_feature_from_wordmap(opts, wordmap)


K = opts.K

hist,_ = np.histogram(wordmap, bins = K)


hist = hist / np.sum(hist)

#######################

K = opts.K
L = opts.L
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
        histogram =  visual_recog.get_feature_from_wordmap(opts, sub_wmaps[i])
        weighted_histogram = histogram * weight
        histogram_all = np.hstack([histogram_all, weighted_histogram])
        # if i==0:
        #     histogram_total = weighted_histogram
        # else:
        #     histogram_total += weighted_histogram
        

# if np.sum(histogram_all) > 0:
#     return histogram_all/np.sum(histogram_all)
# else:
#     return histogram_all

###################


#hist_inter_sim = np.sum(np.minimum(word_hist, histogram), axis = 1)

#return 1-hist_inter_sim
###################


data_dir = opts.data_dir
out_dir = opts.out_dir
SPM_layer_num = opts.L

train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
#img = Image.open(train_files[0])



train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)

dictionary = np.load(join(out_dir, 'dictionary.npy'))

# ----- TODO -----
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
features= []

for training in train_files:
    feat = visual_recog.get_image_feature(opts, training, dictionary)
    features.append(feat)


a = 0
features2 = []
for training in train_files:
    if a < 3:
        feat = visual_recog.get_image_feature(opts, training, dictionary)
        if features2 == []:
            features2 = feat.reshape(1,-1)
            #features2.append(feat) 
        else:
            features2 = np.row_stack([features2, feat])
    else:
        break



path = r"C:\Users\NIshanth Mohankumar\OneDrive\Desktop\CMU_Sem_1\16-720 CV\Assignment_1\HW1\data\laundromat\sun_afrrjykuhhlwiwun.jpg"


prob_image = Image.open(path)
prob_image2 = prob_image.convert('RGB')


if(prob_image.mode != 'RGB'):
    print("hi")
prob_image2.show()
prob_image.show()


norm_image = Image.open(r"C:\Users\NIshanth Mohankumar\OneDrive\Desktop\CMU_Sem_1\16-720 CV\Assignment_1\HW1\data\laundromat\sun_aabvooxzwmzzvwds.jpg")

visual_words.extract_filter_responses(opts, prob_image2)
#############



#dont run this:

# img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
# img = Image.open(img_path)
# #plt.show(img)
# img = np.array(img).astype(np.float32)/255
# plt.imshow(img)
# plt.axis('equal')
# plt.axis('off')
# dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
# wordmap = visual_words.get_visual_words(opts, img, dictionary)
# util.visualize_wordmap(wordmap)


# img_path = join(opts.data_dir, 'park/sun_aiqzpealjtmdbulg.jpg')
# img = Image.open(img_path)
# img = np.array(img).astype(np.float32)/255
# plt.imshow(img)
# plt.axis('equal')
# plt.axis('off')
# dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
# wordmap = visual_words.get_visual_words(opts, img, dictionary)
# util.visualize_wordmap(wordmap)

# img_path = join(opts.data_dir, 'windmill/sun_bedjfgkztzksgrih.jpg')
# img = Image.open(img_path)
# img.show()
# img = np.array(img).astype(np.float32)/255
# plt.imshow(img)
# plt.axis('equal')
# plt.axis('off')
# dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
# wordmap = visual_words.get_visual_words(opts, img, dictionary)
# util.visualize_wordmap(wordmap)



