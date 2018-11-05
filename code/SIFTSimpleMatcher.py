# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 21:36:46 2017

@author: HGY
"""

import numpy as np
from scipy.io import loadmat


#%% SIFTSimpleMatcher function
def SIFTSimpleMatcher(descriptor1, descriptor2, THRESH=0.7):
    '''
    SIFTSimpleMatcher 
    Match one set of SIFT descriptors (descriptor1) to another set of
    descriptors (decriptor2). Each descriptor from descriptor1 can at
    most be matched to one member of descriptor2, but descriptors from
    descriptor2 can be matched more than once.
    
    Matches are determined as follows:
    For each descriptor vector in descriptor1, find the Euclidean distance
    between it and each descriptor vector in descriptor2. If the smallest
    distance is less than thresh*(the next smallest distance), we say that
    the two vectors are a match, and we add the row [d1 index, d2 index] to
    the "match" array.
    
    INPUT:
    - descriptor1: N1 * 128 matrix, each row is a SIFT descriptor.
    - descriptor2: N2 * 128 matrix, each row is a SIFT descriptor.
    - thresh: a given threshold of ratio. Typically 0.7
    
    OUTPUT:
    - Match: N * 2 matrix, each row is a match. For example, Match[k, :] = [i, j] means i-th descriptor in
        descriptor1 is matched to j-th descriptor in descriptor2.
    '''

    #############################################################################
    #                                                                           #
    #                              YOUR CODE HERE                               #
    #                                                                           #
    #############################################################################
    # size
    descriptor1_len = descriptor1.shape[0];
    descriptor2_len = descriptor2.shape[0];
    
    # modify array for matrix operation
    descriptor1_cpy = np.repeat(descriptor1,descriptor2_len,axis=0);
    descriptor2_cpy = np.tile(descriptor2,(descriptor1_len,1));
    
    # calculate distance
    descriptor_dist = descriptor1_cpy - descriptor2_cpy;
    descriptor_dist = np.square(descriptor_dist);
    descriptor_dist_sum = np.sum(descriptor_dist,axis=1);
    descriptor_dist_sum = np.sqrt(descriptor_dist_sum);
    
    # match
    match = np.array([[0,0]]);
    match_num = 0;
    for i in range(descriptor1_len):
        min_dist = np.min(descriptor_dist_sum[i*descriptor2_len : (i+1)*descriptor2_len]);
        min_idx = np.argmin(descriptor_dist_sum[i*descriptor2_len : (i+1)*descriptor2_len]);
        descriptor_dist_sum[i*descriptor2_len + min_idx] = 1000000;
        second_min_dist = np.min(descriptor_dist_sum[i*descriptor2_len : (i+1)*descriptor2_len]);
        if min_dist <= second_min_dist * THRESH:
            if match_num == 0:
                match[match_num,0] = i;
                match[match_num,1] = min_idx;
                match_num = match_num + 1;
            else:
                match = np.append(match,[[i,min_idx]],axis=0);
                
    #############################################################################
    #                                                                           #
    #                             END OF YOUR CODE                              #
    #                                                                           #
    #############################################################################   
    
    return match
