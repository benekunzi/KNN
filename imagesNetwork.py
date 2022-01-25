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

    # small tensor 
    sl = length/32
    sh = height/32

    # preallocate tensor of given shape
    small_batch = torch.zeros(number, int(sl*sh), rgb, 32, 32, dtype=torch.uint8)

    # load images into sensor
    i = 0
    m = 0
    for files in os.listdir(path):
        if ".jpg" in files:
            img_arr = imageio.imread(path + f"/{files}")
            img = torch.from_numpy(img_arr)
            if img.size() == shape_img:
                prop_img = img.permute(2,0,1)
                prop_img = prop_img[:3]         # shape of image 3648*5472*3
                for k in range(0, length, 32):
                    for l in range(0, height, 32):
                        small_batch[i, m] = prop_img[:, k:k+32, l:l+32]
                        m += 1
                        if m == int(sl*sh):
                            m = 0
                i += 1

    # normalizing the data
    small_batch = small_batch.float()
    small_batch /= 255.0

    return small_batch

def loadImagePair(path_orig_img:str, path_edit_img:str):
    # load original image
    orig_img = imageio.imread(path_orig_img)
    orig = torch.from_numpy(orig_img)
    shape_orig_img = orig.size()
    length_orig = shape_orig_img[0]
    height_orig = shape_orig_img[1]
    prop_img = orig.permute(2,0,1)
    prop_img = prop_img[:3]

    # print size of original image
    print(length_orig, height_orig)

    # check if image size can be split up into small 32x32 images
    if length_orig%32 == 0 and height_orig%32 == 0:
        sl_orig = length_orig/32
        sh_orig = height_orig/32
    else:
        raise ValueError("cannot create 32x32x3 image")
    
    # preallocate tensor of given shape
    batch_orig = torch.zeros(int(sl_orig*sh_orig), 3, 32, 32, dtype=torch.uint8)

    # load parts from image into tensor
    k = 0
    for i in range(0, length_orig, 32):
        for j in range(0, height_orig, 32):
            batch_orig[k] = prop_img[:, i:i+32, j:j+32]
            k += 1

    # load edited image
    edit_img = imageio.imread(path_edit_img)
    edit = torch.from_numpy(edit_img)
    shape_edit_img = edit.size()
    length_edit = shape_edit_img[0]
    height_edit = shape_edit_img[1]
    prop_img = edit.permute(2,0,1)
    prop_img = prop_img[:3]

    # print size of edited image
    print(length_edit, height_edit)

    # if they dont have the same size raise error
    if length_orig != length_edit or height_orig != height_edit:
        raise ValueError("images does not have the same size")

    # check if image size can be split up into small 32x32 images
    if length_edit%32 == 0 and height_edit%32 == 0:
        sl_edit = length_edit/32
        sh_edit = height_edit/32
    else:
        raise ValueError("cannot create 32x32x3 image")

    # preallocate tensor of given shape
    batch_edit = torch.zeros(int(sl_edit*sh_edit), 3, 32, 32, dtype=torch.uint8)

    # load parts from image into tensor
    k = 0
    for i in range(0, length_edit, 32):
        for j in range(0, height_edit, 32):
            batch_edit[k] = prop_img[:, i:i+32, j:j+32]
            k += 1

    # normalizing the data
    batch_orig = batch_orig.float()
    batch_orig /= 255.0

    batch_edit = batch_edit.float()
    batch_edit /= 255.0
    
    return (batch_orig, batch_edit)

def loadSingleImage(path):
    # load original image
    image = imageio.imread(path)
    img = torch.from_numpy(image)
    shape_orig_img = img.size()
    print("shape of image: ", shape_orig_img)
    height = shape_orig_img[0]
    length = shape_orig_img[1]
    prop_img = img.permute(2,0,1)
    prop_img = prop_img[:3]
    print("size of changed img: ", prop_img.size())

    # check if image size can be split up into small 32x32 images
    if height%32 == 0 and length%32 == 0:
        sh_orig = height/32
        sl_orig = length/32
    else:
        raise ValueError("cannot create 32x32x3 image")
    
    # preallocate tensor of given shape
    batch = torch.zeros(int(sl_orig*sh_orig), 3, 32, 32, dtype=torch.uint8)

    # load parts from image into tensor
    k = 0
    for i in range(0, height, 32):
        for j in range(0, length, 32):
            batch[k] = prop_img[:, i:i+32, j:j+32]
            k += 1

    batch = batch.float()
    batch /= 255.0

    return batch
    

def gemeinsamerTeiler(zahl1, zahl2):
    gemeinsameTeiler = []
    if zahl1 > zahl2:
        for i in range(1, zahl1):
            if zahl1%i == 0 and zahl2%i == 0:
                gemeinsameTeiler.append(i)
    else:
        for i in range(1, zahl2):
            if zahl1%i == 0 and zahl2%i == 0:
                gemeinsameTeiler.append(i)

    print(gemeinsameTeiler)

def Vielfache32():
    viel = []
    for i in range(1, 200):
        viel.append(32*i)

    print(viel)
