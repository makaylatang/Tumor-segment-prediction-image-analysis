#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 23:53:15 2023

@author: makaylatang
"""

import sys
import os
import numpy as np
import argparse
from fastai.vision.all import *
import fastai
import PIL
import datetime
import warnings
warnings.filterwarnings("ignore")
import pathlib

#if you have a window computer, uncomment the following lines to load model correctly
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

class runTumorSeg:

    def __init__(self):
        self.run_image_size = 500   # this was set by model training parameters
        self.model_path = './model/deeplabv3_resnet50_5ep_1e4lr_wCS.pkl'
        self.image_file = args.image_file

    def procesImage(self):

        #first load pytorch model
        learn = load_learner(self.model_path,cpu=True) #this model runs on CPU, but you can change to cpu-False if you have a GPU

        #now load image and initialize empty mask array for predictions
        image = PIL.Image.open(self.image_file)
        imgid = os.path.split(self.image_file)[1]
        gt_mask = PIL.Image.open('./gt_masks/'+str.replace(imgid,'.tif','.png'))
        output_mask = np.zeros((image.size[1],image.size[0]))

        #TASK 1: THE MODEL RUNS ON IMAGE PATCHES OF SIZE 500x500 PIXELS, YOU WILL NEED TO WRITE CODE THAT
        # (A) SPLITS THE IMAGE INTO APPROPRIATELY SIZED PATCHES FOR INFERENCE
        patch_size = self.run_image_size
        patches = []
        for i in range(0, image.size[0], patch_size):
            for j in range(0, image.size[0], patch_size):
                patch = image.crop((j, i, j + patch_size, i + patch_size))
                patches.append((patch, (j, i)))
                
        # (B) PROPERLY BUILD OUTPUT MASK FROM EACH PATCH OUTPUT
        for patch, (x, y) in patches:
            patch_mask = self.predictPatch(patch=patch, learn=learn)
            output_mask[y : y+patch_size, x : x+patch_size] += patch_mask

        # the resulting output mask should be numpy array of 0s (benign) and 1s (malignant)
        output_mask = output_mask.astype("bool")
        mask_out = PIL.Image.fromarray(output_mask)
        mask_out.save('./ai_masks/'+str.replace(imgid,'.tif','.png'))

        #TASK 2: CALCULATE RELEVANT AI METRICS
        # (A) calculate Dice metric from produced mask and provided ground truth mask
        # Dice = 2TP / (2TP+FP+FN)
        # intersection = TP 
        # should convert image file to array  
        gt_mask_array = np.array(gt_mask)
        TP = np.logical_and(output_mask, gt_mask_array).sum()
        dice = 2 * TP / (np.count_nonzero(output_mask) + np.count_nonzero(gt_mask_array))
        
        # (B) use connected components to calculate Sensitivity and Specificity from the number of TP regions, FP regions, and FN regions
        # calculate the number of False Positive
        FP = np.logical_and(np.logical_not(gt_mask_array), output_mask).sum()
        # calculate the number of False Negative
        FN = np.logical_and(gt_mask_array, np.logical_not(output_mask))
        
        # calculate Sensitivity
        Sensitivity = TP / (TP + FN)
        # calculate Specificity
        Specificity = TP / (TP + FP)

    def predictPatch(self,patch,learn):
        # patch needs to be an numpy array of size 500x500 pixels
        # learn is the model we loaded, no changes necessary to this function
        inp, targ, pred, _ = learn.predict(patch, with_input=True)
        pred_arr = pred.cpu().detach().numpy()
        mask = pred_arr.astype("bool")
        return mask


if __name__ == '__main__':
    # example run commend:
    # python tumorsegment_predict.py --image_file ./images/biopsy_slide-1_5x.tif
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file')
    args = parser.parse_args()
    c = runTumorSeg()
    c.procesImage()


