import imageio
import torch
import os
import matplotlib.pyplot as plt

def loadMultImages(path):
    if "original" in path:
        image = imageio.imread(f"{path}img_5470-o-bearb-zug.jpg")
    else:
        image = imageio.imread(f"{path}img_5470-b-bearb-zug.jpg")

    # get size of horizontal picture
    shape_img = image.shape
    length = shape_img[0]
    height = shape_img[1]
    rgb = shape_img[2]


    # amount of pictures in the directory
    number = 0
    for files in os.listdir(path):
        if ".jpg" in files:
            number += 1

    # preallocate tensor of given shape
    batch = torch.zeros(number, rgb, length, height, dtype=torch.uint8)
    print(batch.size())

    # load images into sensor
    i = 0
    for files in os.listdir(path):
        if ".jpg" in files:
            img_arr = imageio.imread(path + f"/{files}")
            img = torch.from_numpy(img_arr)
            if img.size() == shape_img:
                prop_img = img.permute(2,0,1)
                prop_img = prop_img[:3]         # shape of image 3648*5472*3
                batch[i] = prop_img
                i += 1

    # normalizing the data
    batch = batch.float()
    batch /= 255.0

    return batch

def loadImagePair(path_orig_img:str, path_edit_img:str):
    # load original image
    orig_img = imageio.imread(path_orig_img)
    orig = torch.from_numpy(orig_img)
    prop_img = orig.permute(2,0,1)
    prop_img = prop_img[:3]
    prop_img = prop_img.float()
    prop_img /= 255.0

    # load edited image
    edit_img = imageio.imread(path_edit_img)
    edit = torch.from_numpy(edit_img)
    prop_img_e = edit.permute(2,0,1)
    prop_img_e = prop_img_e[:3]
    prop_img_e = prop_img_e.float()
    prop_img_e /= 255.0
    
    return (prop_img, prop_img_e)

def loadSingleImage(path):
    # load original image
    image = imageio.imread(path)
    img = torch.from_numpy(image)
    prop_img = img.permute(2,0,1)
    prop_img = prop_img[:3]
    print("size of changed img: ", prop_img.size())

    prop_img = prop_img.float()
    prop_img /= 255.0

    return prop_img
