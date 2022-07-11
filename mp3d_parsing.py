import os
import time
import random
import math
import numpy as np
import glob
import torch
import pickle
from PIL import Image, ImageEnhance

import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms

from skimage.filters import gaussian, sobel
from scipy.interpolate import griddata

from numpy import eye 

import cv2

def transfer_color(target, source):
    target = target.astype(float) / 255
    source = source.astype(float) / 255

    target_means = target.mean(0).mean(0)
    target_stds = target.std(0).std(0)

    source_means = source.mean(0).mean(0)
    source_stds = source.std(0).std(0)

    target -= target_means
    target /= target_stds / source_stds
    target += source_means

    target = np.clip(target, 0, 1)
    target = (target * 255).astype(np.uint8)

    return target


def prepare_sizes(inputs, keep_aspect_ratio=False):
    
    height, width, _ = np.array(inputs['left_image']).shape

    if keep_aspect_ratio:
        if feed_height <= height and process_width <= width:
            # can simply crop the image
            target_height = height
            target_width = width

        else:
            # check the constraint
            current_ratio = height / width
            target_ratio = feed_height / process_width

            if current_ratio < target_ratio:
                # height is the constraint
                target_height = feed_height
                target_width = int(feed_height / height * width)

            elif current_ratio > target_ratio:
                # width is the constraint
                target_height = int(process_width / width * height)
                target_width = process_width

            else:
                # ratio is the same - just resize
                target_height = feed_height
                target_width = process_width

    else:
        target_height = feed_height
        target_width = process_width

    inputs = resize_all(inputs, target_height, target_width)
    

    # now do cropping
    if target_height == feed_height and target_width == process_width:
        # we are already at the correct size - no cropping
        pass
    else:
        crop_all(inputs)

    return inputs

def resize_all(inputs, height, width):
    # images
    img_resizer = transforms.Resize(size=(height, width))
    for key in ['left_image', 'background']:
        inputs[key] = img_resizer(inputs[key])
    # disparity - needs rescaling
    disp = inputs['loaded_disparity']
    disp *= width / disp.shape[1]

    disp = cv2.resize(disp.astype(float), (width, height))  # ensure disp is float32 for cv2
    inputs['loaded_disparity'] = disp

    return inputs

def crop_all(inputs):

    # get crop parameters
    height, width, _ = np.array(inputs['left_image']).shape
    top = int(random.random() * (height - feed_height))
    left = int(random.random() * (width - process_width))
    right, bottom = left + process_width, top + feed_height

    for key in ['left_image', 'background']:
        inputs[key] = inputs[key].crop((left, top, right, bottom))
    inputs['loaded_disparity'] = inputs['loaded_disparity'][top:bottom, left:right]

    return inputs



def process_disparity(depth, disable_sharpening=False):
    """ Depth predictions have arbitrary scale - need to convert to a pixel disparity"""
    disparity = depth.copy()
    '''
    disparity = 1.0 / disparity

    # make disparities positive
    min_disp = disparity.min()
    if min_disp < 0:
        disparity += np.abs(min_disp)

    disparity /= disparity.max()  # now 0-1

    disparity *= max_disparity
    '''

    #disparity = 1617.78 / (disparity + 23.33) - 43.33


    if disparity.min() == 0:
        #import pdb; pdb.set_trace()
        disparity[disparity <= 0] = 0.05
    disparity = 5.0 / disparity # max of depth is 5.0

    if not disable_sharpening:
        # now find disparity gradients and set to nearest - stop flying pixels
        edges = sobel(disparity) > 3
        disparity[edges] = 0
        mask = disparity > 0

        try:
            disparity = griddata(np.stack([ys[mask].ravel(), xs[mask].ravel()], 1),
                                 disparity[mask].ravel(), np.stack([ys.ravel(),
                                                                    xs.ravel()], 1),
                                 method='nearest').reshape(feed_height, process_width)
        except (ValueError, IndexError) as e:
            pass  # just return disparity

    return disparity


def get_occlusion_mask(shifted):
    mask_up = shifted > 0
    mask_down = shifted > 0

    shifted_up = np.ceil(shifted)
    shifted_down = np.floor(shifted)

    for col in range(process_width - 2):
        loc = shifted[:, col:col + 1]  # keepdims
        loc_up = np.ceil(loc)
        loc_down = np.floor(loc)

        _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
        (shifted_up[:, col + 2:] != loc_down))).min(-1)
        _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
        (shifted_up[:, col + 2:] != loc_up))).min(-1)

        mask_up[:, col] = mask_up[:, col] * _mask_up
        mask_down[:, col] = mask_down[:, col] * _mask_down

    mask = mask_up + mask_down
    return mask



def project_image(image, disp_map, background_image, disable_background=False):
    image = np.array(image)
    background_image = np.array(background_image)

    # set up for projection
    warped_image = np.zeros_like(image).astype(float)
    warped_image = np.stack([warped_image] * 2, 0)
    pix_locations = xs - disp_map

    # find where occlusions are, and remove from disparity map
    mask = get_occlusion_mask(pix_locations)
    masked_pix_locations = pix_locations * mask - process_width * (1 - mask)

    # do projection - linear interpolate up to 1 pixel away
    weights = np.ones((2, feed_height, process_width)) * 10000

    for col in range(process_width - 1, -1, -1):
        loc = masked_pix_locations[:, col]
        loc_up = np.ceil(loc).astype(int)
        loc_down = np.floor(loc).astype(int)
        weight_up = loc_up - loc
        weight_down = 1 - weight_up

        mask = loc_up >= 0
        mask[mask] = weights[0, np.arange(feed_height)[mask], loc_up[mask]] > weight_up[mask]
        weights[0, np.arange(feed_height)[mask], loc_up[mask]] = weight_up[mask]
        warped_image[0, np.arange(feed_height)[mask], loc_up[mask]] = image[:, col][mask] / 255.

        mask = loc_down >= 0
        mask[mask] = weights[1, np.arange(feed_height)[mask], loc_down[mask]] > weight_down[mask]
        weights[1, np.arange(feed_height)[mask], loc_down[mask]] = weight_down[mask]
        warped_image[1, np.arange(feed_height)[mask], loc_down[mask]] = image[:, col][mask] / 255.

    weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
    weights = np.expand_dims(weights, -1)
    warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
    warped_image *= 255.

    # now fill occluded regions with random background
    if not disable_background:
        warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]

    warped_image = warped_image.astype(np.uint8)

    return warped_image



def parse_all_data(root_path, scenes):
    import pdb; pdb.set_trace()
    data_idx_all = []
    print(root_path)
    with open(root_path, 'rb') as f:
        data_dict = pickle.load(f)
    for scene in scenes:
        print(scene)
        data_idx_all += ['/'.join([scene, str(loc), str(ori)]) \
            for (loc,ori) in list(data_dict[scene].keys())]
        print(len(data_idx_all))    
    
    return data_idx_all, data_dict


#scenes = ['UwV83HsGsw3']
# get scenes of mp3d datasets train/val/test
scenes = []
with open("metadata/mp3d/mp3d_scenes_test.txt", "r") as f:
    for line in f.readlines():
        scenes.append(line[:-1])

max_disparity = 26
feed_width, feed_height = 128, 128
process_width = feed_width + max_disparity
xs, ys = np.meshgrid(np.arange(process_width), np.arange(feed_height))

root_path = './mp3d_dataset/test/'

#data_idx, data = parse_all_data('UwV83HsGsw3.pkl', scenes)

count = 0

total_p = 0.0

for scene in scenes:
    if not os.path.exists(root_path + scene):
        os.makedirs(root_path + scene)
    if not os.path.exists(root_path + scene + '/0/left/'):
        os.makedirs(root_path + scene + '/0/left/')
    if not os.path.exists(root_path + scene + '/0/right/'):
        os.makedirs(root_path + scene + '/0/right/')
    if not os.path.exists(root_path + scene + '/0/depth/'):
        os.makedirs(root_path + scene + '/0/depth/')
    if not os.path.exists(root_path + scene + '/90/left/'):
        os.makedirs(root_path + scene + '/90/left/')
    if not os.path.exists(root_path + scene + '/90/right/'):
        os.makedirs(root_path + scene + '/90/right/')
    if not os.path.exists(root_path + scene + '/90/depth/'):
        os.makedirs(root_path + scene + '/90/depth/')
    if not os.path.exists(root_path + scene + '/180/left/'):
        os.makedirs(root_path + scene + '/180/left/')
    if not os.path.exists(root_path + scene + '/180/right/'):
        os.makedirs(root_path + scene + '/180/right/')
    if not os.path.exists(root_path + scene + '/180/depth/'):
        os.makedirs(root_path + scene + '/180/depth/')
    if not os.path.exists(root_path + scene + '/270/left/'):
        os.makedirs(root_path + scene + '/270/left/')
    if not os.path.exists(root_path + scene + '/270/right/'):
        os.makedirs(root_path + scene + '/270/right/')
    if not os.path.exists(root_path + scene + '/270/depth/'):
        os.makedirs(root_path + scene + '/270/depth/')


    pkl_path = './mp3d_scene_observations/mp3d/' + scene + '.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    for (loc, orn) in list(data.keys()):

        img = Image.fromarray(data[(int(loc),int(orn))][('rgb')]).convert('RGB')
        left_save_path = root_path + scene + '/' + str(orn) + '/left/' + str(loc) + '.png'
        img.save(left_save_path)
        #cv2.imwrite(left_save_path, np.array(img))

        depth = data[(int(loc),int(orn))][('depth')]
        depth_save_path = root_path + scene + '/' + str(orn) + '/depth/' + str(loc) + '.npy'
        np.save(depth_save_path, depth)
        #import pdb; pdb.set_trace()
        #*****  Debug  *******
        aaa = depth <= 5
        total_p += aaa.sum() / (128*128)

            
        background_img = Image.fromarray(data[(int(loc), (int(orn)+90)%360 )][('rgb')]).convert('RGB')
        
        inputs = {}
        inputs['left_image'] = img
        inputs['background'] = background_img
        inputs['loaded_disparity'] = depth

        # resize and/or crop
        inputs = prepare_sizes(inputs)

        inputs['background'] = transfer_color(np.array(inputs['background']), np.array(inputs['left_image']))

        # convert scaleless disparity to pixel disparity
        inputs['disparity'] = process_disparity(inputs['loaded_disparity'])

        # now generate synthetic stereo image
        projection_disparity = inputs['disparity']

        right_image = project_image(inputs['left_image'], projection_disparity, inputs['background'])

        save_right_img = right_image[:, max_disparity:]
        save_right_img = Image.fromarray(save_right_img).convert('RGB')

        right_save_path = root_path + scene + '/' + str(orn) + '/right/' + str(loc) + '.png'
        #cv2.imwrite(right_save_path, save_right_img)
        save_right_img.save(right_save_path)

        print(count, right_save_path)


        count += 1
        #if count > 10:
        #    break

    

#print('depth less than 5:', total_p / count) # <4:90%, <5:94.2%, <6:96.5%

