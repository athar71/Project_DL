#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:09:10 2018
@author: athar & nooshin & Arman
"""
from __future__ import print_function
import os
import glob
import scipy

import tensorflow as tf
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt

#For Dropout
keep_prob_=0.9


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
    "water_bottle":51
    }

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class Dataset(object):
    def __init__(self, rgb_path, depth_path, num_imgs):
        self.RGB_path = rgb_path
        self.Depth_path = depth_path
        self.num_imgs = num_imgs

    def normalize_np_rgb_image(self, image):
        return (image / 255.0 - 0.5) / 0.5

    def normalize_np_depth_image(self, image):
        return (image / 65535.0 - 0.5) / 0.5

    def get_input_rgb(self, image_path):
        image = np.array(Image.open(image_path)).astype(np.float32)
        return self.normalize_np_rgb_image(image)

    def get_input_depth(self, image_path):
        image = np.array(Image.open(image_path)).astype(np.float32)
        return self.normalize_np_depth_image(image)

    def get_imagelist(self, data_path):
        imgs_path = os.path.join(data_path, '*.png')
        all_namelist = glob.glob(imgs_path, recursive=True)
        imgs_name = [f for f in os.listdir(data_path) if f.endswith('.png')]
        return all_namelist[:self.num_imgs], imgs_name

    def get_label(self, STR):
        str1 = STR.split('_',1)[0]
        str2 = STR.split('_',1)[1]
        if not RepresentsInt(str2):
            str1 = str1 + '_' + str2
        label = Label_dict[str1]
        return label

    def get_nextbatch(self, batch_size):
        assert (batch_size > 0),"Give a valid batch size"
        cur_idx = 0
        image_namelist, image_names = self.get_imagelist(self.RGB_path)
        while cur_idx + batch_size <= self.num_imgs:
            cur_namelist_rgb = image_namelist[cur_idx:cur_idx + batch_size]
            cur_namelist_depth = [os.path.join(Depth_path,depth_image) for depth_image in image_names]
            cur_batch_rgb = [self.get_input_rgb(image_path) for image_path in cur_namelist_rgb]
            cur_batch_rgb = np.array(cur_batch).astype(np.float32)
            cur_batch_depth = [self.get_input_depth(image_path) for image_path in cur_namelist_depth]
            cur_batch_depth = np.array(cur_batch).astype(np.float32)
            cur_idx += batch_size
            labels = [self.get_label() for name in image_names]
            yield cur_batch_rgb, cur_batch_depth, labels

    def get_nextbatch_RGBonly(self, batch_size):
        assert (batch_size > 0),"Give a valid batch size"
        cur_idx = 0
        image_namelist, image_names = self.get_imagelist(self.RGB_path)
        while cur_idx + batch_size <= self.num_imgs:
            cur_namelist_rgb = image_namelist[cur_idx:cur_idx + batch_size]
            cur_batch_rgb = [self.get_input_rgb(image_path) for image_path in cur_namelist_rgb]
            cur_batch_rgb = np.array(cur_batch).astype(np.float32)
            cur_idx += batch_size
            labels = [self.get_label() for name in image_names]
            yield cur_batch_rgb, labels

    def show_image(self, image, normalized=True):
        if not type(image).__module__ == np.__name__:
            image = image.numpy()
        if normalized:
            npimg = (image * 0.5) + 0.5
        npimg.astype(np.uint8)
        plt.imshow(npimg, interpolation='nearest')





def rgbd_dataset_generator(dataset_name, batch_size):
    pass
#    assert dataset_name in ['train', 'test']
#    assert batch_size > 0 or batch_size == -1  # -1 for entire dataset
#    
#    path = './svhn_mat/' # path to the SVHN dataset you will download in Q1.1
#    file_name = '%s_32x32.mat' % dataset_name
#    file_dict = scipy.io.loadmat(os.path.join(path, file_name))
#    X_all = file_dict['X'].transpose((3, 0, 1, 2))
#    y_all = file_dict['y']
#    data_len = X_all.shape[0]
#    batch_size = batch_size if batch_size > 0 else data_len
#    
#    X_all_padded = np.concatenate([X_all, X_all[:batch_size]], axis=0)
#    y_all_padded = np.concatenate([y_all, y_all[:batch_size]], axis=0)
#    y_all_padded[y_all_padded == 10] = 0
#    
#    for slice_i in range(int(math.ceil(data_len / batch_size))):
#        idx = slice_i * batch_size
#        X_batch = X_all_padded[idx:idx + batch_size]
#        y_batch = np.ravel(y_all_padded[idx:idx + batch_size])
#        yield X_batch, y_batch
    
def mdl_rgb_d(x_rbg,x_depth):
    
    """
    First we define the stram for the rgb images
    """
    
    convR1 = tf.layers.conv2d(
            inputs=x_rbg,
            filters= 96,  # number of filters, Integer, the dimensionality of the output space 
            strides= 4, # convolution stride
            kernel_size=[11, 11],
            padding="valid",
            activation=tf.nn.relu)
    
   poolR1 = tf.layers.max_pooling2d(inputs=convR1, 
                                    pool_size=[3, 3], 
                                    strides=2)  # strides of the pooling operation 
    
   convR2 = tf.layers.conv2d(
            inputs = poolR1,
            filters = 256, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [5,5],
            padding="valid",
            activation=tf.nn.relu)
    
   poolR2 = tf.layers.max_pooling2d(inputs=convR2, 
                                    pool_size=[3, 3], 
                                    strides = 2)   # strides of the pooling operation 
    
   convR3 = tf.layers.conv2d(
            inputs = poolR2,
            filters = 384, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
    
   convR4 = tf.layers.conv2d(
            inputs = convR3,
            filters = 384, # number of filters
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
   
    
   convR5 = tf.layers.conv2d(
            inputs = convR4,
            filters = 256, # number of filters
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
   poolR5 = tf.layers.max_pooling2d(inputs=convR5, 
                                    pool_size=[3, 3], 
                                    strides=2)   # strides of the pooling operation 
   
   
#   fcR6 =  tf.contrib.layers.fully_connected (
#            inputs = poolD5,
#            num_outputs = 4096,
#            activation_fn=tf.nn.relu)
#   
#   fcR7 = tf.contrib.layers.fully_connected (
#            inputs = fcD6,
#            num_outputs = 4096,
#            activation_fn=tf.nn.relu)
   
   pool_flatR = tf.contrib.layers.flatten(poolR5, scope='pool2flat')
   fcR6 = tf.layers.dense(inputs=pool_flatR, units=4096, activation=tf.nn.relu)
   fcR6Drop=tf.nn.dropout(fcR6,keep_prob_)
   fcR7 = tf.layers.dense(inputs=fcR6Drop, units=4096, activation=tf.nn.relu)
   fcR7Drop=tf.nn.dropout(fcR7,keep_prob_)
    
   """
     define the stram for the depth images
     
    """    
    
    convD1 = tf.layers.conv2d(
            inputs=x_depth,
            filters= 96,  # number of filters, Integer, the dimensionality of the output space 
            strides= 4, # convolution stride
            kernel_size=[11, 11],
            padding="valid",
            activation=tf.nn.relu)
    
   poolD1 = tf.layers.max_pooling2d(inputs=convD1, 
                                    pool_size=[3, 3], 
                                    strides=2)  # strides of the pooling operation 
    
   convD2 = tf.layers.conv2d(
            inputs = poolD1,
            filters = 256, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [5,5],
            padding="valid",
            activation=tf.nn.relu)
    
   poolD2 = tf.layers.max_pooling2d(inputs=convD2, 
                                    pool_size=[3, 3], 
                                    strides = 2)   # strides of the pooling operation 
    
   convD3 = tf.layers.conv2d(
            inputs = poolD2,
            filters = 384, # number of filters, Integer, the dimensionality of the output space 
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
    
   convD4 = tf.layers.conv2d(
            inputs = convD3,
            filters = 384, # number of filters
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
   
    
   convD5 = tf.layers.conv2d(
            inputs = convD4,
            filters = 256, # number of filters
            kernel_size = [3,3],
            padding="valid",
            activation=tf.nn.relu)
    
   poolD5 = tf.layers.max_pooling2d(inputs=convD5, 
                                    pool_size=[3, 3], 
                                    strides=2)   # strides of the pooling operation 
   
   
#   fcD6 =  tf.contrib.layers.fully_connected (
#            inputs = poolD5,
#            num_outputs = 4096,
#            activation_fn=tf.nn.relu)
#   
#   fcD7 = tf.contrib.layers.fully_connected (
#            inputs = fcD6,
#            num_outputs = 4096,
#            activation_fn=tf.nn.relu)
   
   pool_flatD = tf.contrib.layers.flatten(poolD5, scope='pool2flat')
   fcD6 = tf.layers.dense(inputs=pool_flatD, units=4096, activation=tf.nn.relu)
   fcD6Drop=tf.nn.dropout(fcD6,keep_prob_)
   fcD7 = tf.layers.dense(inputs=fcD6, units=4096, activation=tf.nn.relu)
   fcD7Drop=tf.nn.dropout(fcD7,keep_prob_)
   """
   fc8 = tf.contrib.layers.fully_connected (
            inputs = tf.concat((fcR7, fcD7), axis=1),
            num_outputs = 4096,
            activation_fn=tf.nn.relu)
   
   fc9 = tf.contrib.layers.fully_connected (
            inputs = fc8,
            num_outputs = 51,
            activation_fn=tf.nn.relu)
   
   """
   fc8 = tf.layers.dense(inputs=tf.concat((fcR7Drop, fcD7Drop), axis=1), units=4096, activation=tf.nn.relu)
   fc9 = tf.layers.dense(inputs=fc8, units=51)
   
   
   """
   pool_flat = tf.contrib.layers.flatten(pool2, scope='pool2flat')
   dense = tf.layers.dense(inputs=pool_flat, units=500, activation=tf.nn.relu)
   logits = tf.layers.dense(inputs=dense, units=10)
   """
   
   return fc9



def apply_classification_loss(model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):  # use gpu:0 if on GPU
            x_rgb = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x_rgb')
            x_depth = tf.placeholder(tf.float32, [None, 227, 227, 1], name='x_depth')
            y_ = tf.placeholder(tf.int32, [None], name='y_')
            y_logits = model_function(x_rgb, x_depth)
            
            y_dict = dict(labels=y_, logits=y_logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(**y_dict)
            cross_entropy_loss = tf.reduce_mean(losses)
            train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy_loss)
           
            
            y_pred = tf.argmax(tf.nn.softmax(y_logits), axis=1)
            correct_prediction = tf.equal(tf.cast(y_pred, tf.int32), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    model_dict = {'graph': g, 'inputs': [x_rgb,x_depth, y_], 'train_op': train_op,
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
#This is the general idea but needs to be modified                    
data_loader_train = Dataset(rgb_path_train, depth_path_train, num_imgs_train)            
data_loader_test = Dataset(rgb_path_test, depth_path_test, num_imgs_test)           
dataset_generators = {
        'train': Dateset.get_nextbatch_RGBonly(data_loader_train, 256),
        'test': Dateset.get_nextbatch_RGBonly(data_loader_test, 256),
}
    
model_dict = apply_classification_loss(mdl_rgb_d)
train_model(model_dict, dataset_generators, epoch_n=50, print_every=20)    


