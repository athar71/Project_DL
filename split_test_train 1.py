# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 21:07:58 2018

@author: megbe
"""

import argparse
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument('file_list') # rgb file list
parser.add_argument('depth_file_list') #rgbdepth file list
args = parser.parse_args()


def get_objects(file_list):    
    objects = [] 
    file_dict = []    
    for x in file_list:
        orig_file_name = x.strip()
        x = x.strip().split('_')        
        if x[1].isdigit():
            object_name = x[0]
        else:
            object_name = "_".join([x[0], x[1]])            
        objects.append(object_name)
        file_dict.append({'file_name':orig_file_name, 'object':object_name})
    objects = list(set(objects))
    return objects, file_dict

def create_master_dict(objects, file_list, TRAIN_RATIO=0.8):    
    MASTER_DICT = {}
    for obj in objects:        
        object_matches = []
        for file in file_dict:                        
            if obj == file['object']:
                object_matches.append(file['file_name'])            
            MASTER_DICT.update({obj:object_matches})    
    return MASTER_DICT

def split_rgb(MASTER_DICT):
    TRAIN_FILES = []; TEST_FILES = [];
    for obj in objects:
        num_pics = len(MASTER_DICT[obj])
        pics_idx = list(range(num_pics))
        random.shuffle(pics_idx, random.random)        
        len_train = int(TRAIN_RATIO*num_pics)
        train_set = pics_idx[:len_train]
        test_set = pics_idx[len_train+1:]
        train_files = [MASTER_DICT[obj][x] for x in train_set]
        test_files = [MASTER_DICT[obj][x] for x in test_set]        
        for x in train_files:
            TRAIN_FILES.append(x)
        for x in test_files:
            TEST_FILES.append(x)
    return TRAIN_FILES, TEST_FILES

def match_depth(TRAIN_FILES, TEST_FILES, depth_file_list):
    
    rgb_train = []; rgb_test = [];
    for x in TRAIN_FILES:
        rgb_train.append("_".join(x.split('_')[:-2]))
    for x in TEST_FILES:
        rgb_test.append("_".join(x.split('_')[:-2]))
    
    depth_TRAIN_FILES = []; depth_TEST_files = [];
    for dfile in depth_file_list:

        uid = dfile.strip().split('_')[:-2]
        uid = "_".join(uid)
        if uid in rgb_train:
            depth_TRAIN_FILES.append(dfile.strip())
#            print('uid train', uid)
        elif uid in rgb_test:
            depth_TEST_files.append(dfile.strip())
#            print('uid test', uid)
            
    return depth_TRAIN_FILES, depth_TEST_files

if __name__ == "__main__":
    
    with open(args.file_list, 'r') as file_list:
        file_list = file_list.readlines()

    with open(args.depth_file_list, 'r') as depth_file_list:
        depth_file_list = depth_file_list.readlines()
        
    objects, file_dict = get_objects(file_list)
    
    print('objects', objects)
    
    TRAIN_RATIO = 0.8
    
    MASTER_DICT = create_master_dict(objects, file_list, TRAIN_RATIO)

    TRAIN_FILES, TEST_FILES = split_rgb(MASTER_DICT) 
    
    print('TRAIN RATIO: ', TRAIN_RATIO, 'TRAIN=', len(TRAIN_FILES), 'TEST=', len(TEST_FILES))                    

    depth_TRAIN_FILES, depth_TEST_FILES = match_depth(TRAIN_FILES, TEST_FILES, depth_file_list)

    print('TRAIN RATIO: ', TRAIN_RATIO, 'TRAIN=', len(depth_TRAIN_FILES), 'TEST=', len(depth_TEST_FILES))                    
    
    with open(args.file_list[:-4]+'_TRAIN_FILES.txt', 'w') as train_output:
        for line in TRAIN_FILES:
            train_output.write(line)
            train_output.write('\n')
            
    with open(args.file_list[:-4]+'_TEST_FILES.txt', 'w') as test_output:
        for line in TEST_FILES:
            test_output.write(line)
            test_output.write('\n')
        
    with open(args.file_list[:-4]+'_DEPTH_TRAIN_FILES.txt', 'w') as dtrain_output:
        for line in depth_TRAIN_FILES:
            dtrain_output.write(line)
            dtrain_output.write('\n')
    
    with open(args.file_list[:-4]+'_DEPTH_TEST_FILES.txt', 'w') as dtest_output:
        for line in depth_TEST_FILES:
            dtest_output.write(line)
            dtest_output.write('\n')
    
    
    
    
    
    
    
    