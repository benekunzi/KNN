import cv2
import os
import torch

# img = cv2.imread('Bilder/Bilder original zugeschnitten/img_5470-o-bearb-zug.jpg')

# scale = 0.25
# width = int(img.shape[1] * scale)
# height = int(img.shape[0] * scale)

# resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
# gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

# cv2.imshow('original image', img)
# cv2.imshow('resized image', resized_img)
# cv2.imshow('grayscale', gray)

# print(img.shape)
# print(resized_img.shape)

# cv2.waitKey(0) 

def loadMultImages(path, grayscale=False):
    i = 0
    for files in os.listdir(path):
        if ".jpg" in files:
            i += 1

    print("Number of images in path: ", i)

    if grayscale:
        batch = torch.zeros(i, 1144, 752)
    else:
        batch = torch.zeros(i,3,1144,752)
    i = 0
    for files in os.listdir(path):
        if ".jpg" in files:
            img = cv2.imread(path + f"/{files}")            # returns numpy array in BGR not RGB!!!
            img_resized = cv2.resize(img, (752, 1144), interpolation=cv2.INTER_AREA)
            if grayscale:
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                tens = torch.from_numpy(img_gray)
                batch[i] = tens
                i += 1
            else:
                tens = torch.from_numpy(img_resized)
                prop_tens = tens.permute(2,0,1)
                prop_tens = prop_tens[:3]
                batch[i] = prop_tens
                i += 1

    batch = batch.float()
    batch /= 255.0

    return batch


def loadSingleImage(path, grayscale=False):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (752, 1144), interpolation=cv2.INTER_AREA)
    if grayscale:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        prop_tens = torch.from_numpy(img_gray)
    else:
        tens = torch.from_numpy(img_resized)
        prop_tens = tens.permute(2,0,1)
        prop_tens = prop_tens[:3]

    prop_tens = prop_tens.float()
    prop_tens /= 255.0

    return prop_tens