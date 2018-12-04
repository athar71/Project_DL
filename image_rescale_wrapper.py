#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:44:12 2018

@author: megbert
"""

import argparse
import os
import subprocess

# location of the image_border.py script
IMAGE_BORDER = '/projectnb/dl-course/MANTA/SCRIPTS/image_rescale.py'
#SAVE_DIR = '/projectnb/dl-course/MANTA/prepped-rgbd-dataset/rgb_rescale_images'

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('file_list')
    parser.add_argument('save_dir')
    args = parser.parse_args()
    
    with open(args.file_list, 'r') as file_list:
        file_list = file_list.readlines()
        
    for file in file_list:
        file = file.strip()

        cmd = ['python', IMAGE_BORDER, '--', file, args.save_dir]
        #print(" ".join(cmd))
        subprocess.call(cmd)
    