#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:50:36 2018

@author: athar
"""
from __future__ import print_function

import math 

import tensorflow as tf
import numpy as np
from PIL import Image



Label_dict = {
    "apple":1,
    "ball":2,
    "banana":3,
    "bell_pepper":4,
    "binder":5,
    "bowl":6,
    "calculator":7,
    "camera":8,
    "cap":9,
    "cell_phone":10,
    "cereal_box":11,
    "coffee_mug":12,
    "comb":13,
    "dry_battery":14,
    "flashlight":15,
    "food_bag":16,
    "food_box":17,
    "food_can":18,
    "food_cup":19,
    "food_jar":20,
    "garlic":21,
    "glue_stick":22,
    "greens":23,
    "hand_towel":24,
    "instant_noodles":25,
    "keyboard":26,
    "kleenex":27,
    "lemon":28,
    "lightbulb":29,
    "lime":30,
    "marker":31,
    "mushroom":32,
    "notebook":33,
    "onion":34,
    "orange":35,
    "peach":36,
    "pear":37,
    "pitcher":38,
    "plate":39,
    "pliers":40,
    "potato":41,
    "rubber_eraser":42,
    "scissors":43,
    "shampoo":44,
    "soda_can":45,
    "sponge":46,
    "stapler":47,
    "tomato":48,
    "toothbrush":49,
    "toothpaste":50,
    "water_bottle":51}


def normalize_np_rgb_image(image):
        return (image / 255.0 - 0.5) / 0.5

def get_input_rgb(image_path):
        image = np.array(Image.open(image_path)).astype(np.float32)
        return normalize_np_rgb_image(image)
    
def get_objects_label(file_list):
    
        objects = [] 
        label = []    
        for x in file_list:
            x = x.strip().split('_')
            
            if x[1].isdigit():
                object_name = x[0]
            else:
                object_name = "_".join([x[0], x[1]])
                
            objects.append(object_name)
            label.append(Label_dict[object_name])
    
        objects = list(set(objects))
    
        return objects, label   
    
def rgb_dataset_generator(file, image_add, batch_size):
    
    assert batch_size > 0 or batch_size == -1  # -1 for entire dataset
    with open(file, 'r') as file_list:
        image_namelist = file_list.readlines()
                    
    X_all = [get_input_rgb(image_add+'/'+image_path[:-1]) for image_path in image_namelist]
    X_all = np.array(X_all).astype(np.float32)
    _, y_all = get_objects_label(image_namelist)
    data_len = X_all.shape[0]
    batch_size = batch_size if batch_size > 0 else data_len
    
    X_all_padded = np.concatenate([X_all, X_all[:batch_size]], axis=0)
    y_all_padded = np.concatenate([y_all, y_all[:batch_size]], axis=0)
    #y_all_padded[y_all_padded == 10] = 0
    
    for slice_i in range(int(math.ceil(data_len / batch_size))):
        idx = slice_i * batch_size
        X_batch = X_all_padded[idx:idx + batch_size]
        y_batch = np.ravel(y_all_padded[idx:idx + batch_size])
        yield X_batch, y_batch    
        
def mdl_rgbonly(x_rbg):
    
    keep_prob_=0.9
    
    convR1 = tf.layers.conv2d(
            inputs=x_rbg,
            filters= 96,  # number of filters, Integer, the dimensionality of the output space 
            strides= 4, # convolution stride
            kernel_size=[11, 11],
            padding="same",
            activation=tf.nn.relu)
    
    poolR1 = tf.layers.max_pooling2d(inputs=convR1, 
                                    pool_size=[3, 3],
                                    strides=2,
                                    padding='valid')  # strides of the pooling operation 
    
    convR2 = tf.layers.conv2d(
            inputs = poolR1,
            filters = 256, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [5,5],
            padding="same",
            activation=tf.nn.relu)
    
    poolR2 = tf.layers.max_pooling2d(inputs=convR2, 
                                    pool_size=[3, 3], 
                                    strides = 2,
                                    padding='valid')   # strides of the pooling operation 
    
    convR3 = tf.layers.conv2d(
            inputs = poolR2,
            filters = 384, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [3,3],
            padding="same",
            activation=tf.nn.relu)
    
    
    convR4 = tf.layers.conv2d(
            inputs = convR3,
            filters = 384, # number of filters
            kernel_size = [3,3],
            padding="same",
            activation=tf.nn.relu)
    
   
    
    convR5 = tf.layers.conv2d(
            inputs = convR4,
            filters = 256, # number of filters
            kernel_size = [3,3],
            padding="same",
            activation=tf.nn.relu)
    
    poolR5 = tf.layers.max_pooling2d(inputs=convR5, 
                                    pool_size=[3, 3], 
                                    strides=2,
                                    padding='valid')    # strides of the pooling operation 
   

    pool_flatR = tf.contrib.layers.flatten(poolR5, scope='pool2flat')
    fcR6 = tf.layers.dense(inputs=pool_flatR, units=4096, activation=tf.nn.relu)
    fcR6Drop=tf.nn.dropout(fcR6,keep_prob_)
    fcR7 = tf.layers.dense(inputs=fcR6Drop, units=4096, activation=tf.nn.relu)
    fcR7Drop=tf.nn.dropout(fcR7,keep_prob_)
    
    fc9 = tf.layers.dense(inputs=fcR7Drop, units=51)
   
    return fc9


def apply_classification_loss(model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/cpu:0"):  # use gpu:0 if on GPU
            x_ = tf.placeholder(tf.float32, [None, 227, 227, 3])
            y_ = tf.placeholder(tf.int32, [None])
            y_logits = model_function(x_)
            
            y_dict = dict(labels=y_, logits=y_logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(**y_dict)
            cross_entropy_loss = tf.reduce_mean(losses)
            trainer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = trainer.minimize(cross_entropy_loss)
            
            y_pred = tf.argmax(tf.nn.softmax(y_logits), axis=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    model_dict = {'graph': g, 'inputs': [x_, y_], 'train_op': train_op,
                  'accuracy': accuracy, 'loss': cross_entropy_loss}
    
    return model_dict

def train_model(model_dict, dataset_generators, epoch_n, print_every):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch_i in range(epoch_n):
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                sess.run(model_dict['train_op'], feed_dict=train_feed_dict)
                
                if iter_i % print_every == 0:
                    collect_arr = []
                    for test_batch in dataset_generators['test']:
                        test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                        to_compute = [model_dict['loss'], model_dict['accuracy']]
                        collect_arr.append(sess.run(to_compute, test_feed_dict))
                    averages = np.mean(collect_arr, axis=0)
                    fmt = (epoch_i, iter_i, ) + tuple(averages)
                    print('epoch {:d} iter {:d}, loss: {:.3f}, '
                          'accuracy: {:.3f}'.format(*fmt))





                   
rgb_path_train='/projectnb/dl-course/MANTA/prepped-rgbd-dataset/rgb_filelist_TRAIN_FILES.txt'
rgb_path_test='/projectnb/dl-course/MANTA/prepped-rgbd-dataset/rgb_filelist_TEST_FILES.txt' 

rgbimagepath ='/projectnb/dl-course/MANTA/prepped-rgbd-dataset/rgb_rescale_images'                    

       
dataset_generators = {
        'train': rgb_dataset_generator(rgb_path_train, rgbimagepath,10),
        'test': rgb_dataset_generator(rgb_path_test,rgbimagepath, 10),
}
    
model_dict = apply_classification_loss(mdl_rgbonly)
train_model(model_dict, dataset_generators, epoch_n=2, print_every=20)                    