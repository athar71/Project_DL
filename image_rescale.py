#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:15:51 2018

@author: taylerpauls
"""

from PIL import Image as PImage
from PIL import ImageOps
from PIL import ImageColor
import numpy as np
import argparse
import os
#import png
import cv2

   
def update_pixels(img, specific_pv):
    
    pixels = img.load() 
    
    for i in range(img.size[1]):    # for every row:
    #change each col for row 1
    # get new value of pixels for each row
    # pixels change for each row
        pv = specific_pv[i]
        b = pv[0]
        h = pv[1]
        k = pv[2]
        for j in range(img.size[0]):    # For every row
        # the vals of pv do not change just update all row vals
            pixels[j,i] = (b, h, k) # set the colour accordingly
    
    return img


def update_pixels_depth(img, specific_pv):
    
    pixels = img.load() 
    
    print('specific_pv', specific_pv) 
    print('img size', img.size)
    
    for i in range(img.size[1]):
        pv = specific_pv[i]
#        pixels[:,i] = pv
        for j in range(img.size[0]):
            pixels[j,i] = pv
    return img
    
def image_scale(image, depth=False):
    print('depth == {}'.format(depth))
    im = image
#    im.show()
    width = im.size[0];
    height = im.size[1];
    print('original image size (w x h): ', im.size)
    max_size = 277;

    if(width > height):
        # calculate new dimensions, while maintaining aspect ratio:         
        new_height_dim = int((height*max_size)/width)
        new_width_dim = max_size        
        im = im.resize((max_size,new_height_dim))
        print('new image size (w x h): ', im.size)
        print("width bigger than height, add height on top and bottom")

        # height of image needs to change
        pix_val = list(im.getdata()) # left to right starting top left corner
        top_b = pix_val[0:new_width_dim] # this is for if row is bigger than column
        bottom_b = pix_val[new_height_dim*new_width_dim-new_width_dim:new_width_dim*new_height_dim]
        print('pixel length (top, bottom): ', len(top_b), len(bottom_b)) # both of these are 84
        
        add_height_dim = max_size-im.size[1]
        top_height_dim = int(add_height_dim/2)
        if add_height_dim % 2 == 0:
            bottom_height_dim = int(add_height_dim/2)
        elif add_height_dim % 2 == 1:
            bottom_height_dim = int(add_height_dim/2) + 1

        print("add height dim: ", top_height_dim, bottom_height_dim)

        # Make image for the "TOP" - if total is an odd number, the top dim will be smaller than the bottom dim        
        img = PImage.new( 'RGB', (im.size[0],top_height_dim), "black") # create a new black image
        # "Make image for the "BOTTOM" - if total is an odd number, the bottom will be larger than the top                
        img2 = PImage.new( 'RGB', (im.size[0],bottom_height_dim), "black") # create a new black image
        
        if depth == False:
            img = update_pixels(img, top_b)
            img2 = update_pixels(img2, bottom_b)            
        elif depth == True:
            img = update_pixels_depth(img, top_b)
            img2 = update_pixels_depth(img2, bottom_b)
                        
        # now append top_image to have different border values :
        new_width = im.size[0]    
        new_height = im.size[1]+img.size[1]+img2.size[1]
        new_im = PImage.new('RGB',(new_width,new_height))
        new_im.paste(img,(0,0)) # top
        new_im.paste(im,(0,img.size[1])) # middle 
        new_im.paste(img2,(0,img.size[1]+im.size[1])) # bottom
#        new_im.show()
        print('final image size (w x h): ', new_im.size)
    
    
    elif(height > width): 
        # calculate new dimensions, while maintaining aspect ratio: 
        new_width_dim = int((width*max_size)/height)
        new_height_dim = max_size
        im = im.resize((new_width_dim,max_size))
        print('new image size (w x h): ', im.size)
        print("width smaller than height, add left and right")

        # width of the image needs to change
        pix_val = list(im.getdata()) # left to right starting top left corner
        left_b = []
        for k in range(new_height_dim):
            ind = k*new_width_dim
            left_b.append(pix_val[ind])
        
        right_b = []
        for k in range(new_height_dim):
            ind = k*new_width_dim-1
            right_b.append(pix_val[ind])
        
        print('pixel length (left, right): ', len(left_b), len(right_b)) # both of these are 84
        
        add_width_dim = max_size-im.size[0]
        if add_width_dim % 2 == 0:
            left_width_dim = int(add_width_dim/2)
            right_width_dim = int(add_width_dim/2)
        elif add_width_dim % 2 == 1:
            left_width_dim = int(add_width_dim/2)
            right_width_dim = int(add_width_dim/2) + 1
            
        print("add width dim: ", left_width_dim, right_width_dim)
        
        # now make two different images one for left and one for right 
        # update the pixels to left_b and right_b

        # Make image for "LEFT" - if total is an odd number, left side will be one smaller than right
        img = PImage.new( 'RGB', (left_width_dim,im.size[1]), "black") # create a new black image              
        # "Make image for right" - if total is an odd number, right side will be one larger than left        
        img2 = PImage.new( 'RGB', (right_width_dim,im.size[1]), "black") # create a new black image
        
        if depth == False:
            img = update_pixels(img, left_b)
            img2 = update_pixels(img, right_b)
        elif depth == True:
            img = update_pixels_depth(img, left_b)
            img2 = update_pixels_depth(img, right_b)
                            
        # now append top_image to have different border values :
        new_width = im.size[0]+img.size[0]+img2.size[0]
        new_height = im.size[1]
        new_im = PImage.new('RGB',(new_width,new_height))
        new_im.paste(img,(0,0))#top
        new_im.paste(im,(img.size[0],0))
        new_im.paste(img2,(img.size[0]+im.size[0],0))
        #new_im.show()
        print('final image size (w x h): ', new_im.size)
        
        
    elif(height == width):
        
        new_im = im.resize((max_size,max_size))        
        
    return new_im
        

def depth_image_scale(image):
    
    ## This function uses an "image" which was opened in cv2
    
    print("image shape (h x w): ", image.shape)    
    height, width = image.shape    
    max_size = 277;

    if height > width:
        print('height is bigger than width, need to add to left and right')

        new_height = max_size
        new_width = int(width * (max_size / height))
        
        res_image = cv2.resize(image, (new_width, new_height))
        
        print('new shape: ', res_image.shape)

        width_diff = max_size-new_width
        left_width = int(np.divide(width_diff, 2))

        if width_diff % 2 == 1:
            right_width = int(np.divide(width_diff, 2)) + 1
        elif width_diff % 2 == 0:
            right_width = int(np.divide(width_diff, 2))
        
#        print('top, bottom', top_height, bottom_height)
        
        new_image = cv2.copyMakeBorder(res_image, 0, 0, right_width, left_width, cv2.BORDER_REPLICATE)
                
        print('final shape: ', new_image.shape)

#        cv2.imshow('Image', image)
#        cv2.waitKey(0) & 0xFF
#        cv2.destroyAllWindows()

    elif height < width:
        print('height is smaller than width, need to add to top and bottom')
        
        new_width = max_size
        new_height = int(height * (max_size / width))
                
        res_image = cv2.resize(image, (new_width, new_height))
        
        print('new shape: ', res_image.shape)
        
        height_diff = max_size-new_height
        top_height = int(np.divide(height_diff, 2))

        if height_diff % 2 == 1:
            bottom_height = int(np.divide(height_diff, 2)) + 1
        elif height_diff % 2 == 0:
            bottom_height = int(np.divide(height_diff, 2))
        
#        print('top, bottom', top_height, bottom_height)
        
        new_image = cv2.copyMakeBorder(res_image, top_height, bottom_height, 0, 0, cv2.BORDER_REPLICATE)
                
        print('final shape: ', new_image.shape)
        
    elif height == width:
        
        new_image = cv2.resize(image, (max_size, max_size))
        
        print('new shape: ', new_image.shape)
        

    return new_image    
        
#    cv2.imshow('Image', image)
#    cv2.waitKey(0) & 0xFF
#    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('save_path')
    args = parser.parse_args()    
                    
    # The pixel update is different for rgb versus depth images, need to specify
    image_file_name = os.path.split(args.image)[1]        

    if 'depth' not in image_file_name:
        depth = False
    
        image = PImage.open(args.image)
        new_image = image_scale(image)    

        new_image.save(os.path.join(args.save_path, os.path.split(args.image)[1][:-4]+'_rescale.png'))


    elif 'depth' in image_file_name:
        depth = True

        image = cv2.imread(args.image, -1)
        
        new_image = depth_image_scale(image)

        cv2.imwrite(os.path.join(args.save_path, os.path.split(args.image)[1][:-4]+'_rescale.png'), new_image)

#        image = PImage.open(args.image)
#        pixel = np.array(image)
#        print("PIL:", pixel.dtype)
#        print(max(max(row) for row in pixel))
    
#        reader = png.Reader(args.image)
#        pngdata = reader.read()
#        px_array = np.array(map(np.uint16, pngdata[2]))
#        print(px_array.dtype)

# Tayler testing code
#    image_path_ball = "/Users/taylerpauls/Desktop/ball/ball/ball_1/ball_1_1_1_crop.png"
#    image_path_apple = "/Users/taylerpauls/Desktop/apple/apple/apple_1/apple_1_1_1_crop.png"
#    im_b = PImage.open(image_path_ball)
#    im_a = PImage.open(image_path_apple)
#    new_a = image_scale(im_a)    
#    new_b = image_scale(im_b)
        
        
 
    