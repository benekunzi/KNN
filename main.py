import sys
import neural_network
import torch
import torch.nn.functional as F
from progressBar import ProgressBar
import torch.optim as optim
from unsorting import unsorting
# from fullImageLoader import loadMultImages
import os 
import numpy as np
import matplotlib.pyplot as plt
from downsize import loadMultImages

if __name__ == "__main__":
    grayscale = True
    orig = loadMultImages("Bilder/Bilder original zugeschnitten", grayscale)
    edit = loadMultImages("Bilder/Bilder bearbeitet zugeschnitten", grayscale)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # for mlp
    net = neural_network.Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    EPOCHS = 2

    progressbar = ProgressBar(len(orig))

    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
        print('-' * 22)
        for i in range(len(orig)):
            progressbar.progressBar(i)
            if grayscale:
                orig_flatten = torch.flatten(orig[i])
                edit_flatten = torch.flatten(edit[i])
                output = net(orig_flatten)
                loss = F.mse_loss(output, edit_flatten)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            else:
                for j in range(3):
                    orig_flatten = torch.flatten(orig[i, j, :, :])
                    edit_flatten = torch.flatten(edit[i, j, :, :])
                    output = net(orig_flatten)
                    loss = F.mse_loss(output, edit_flatten)
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()
        print("Loss: ", loss)
    torch.save(net.state_dict(), "Model_saved/model.pth")   

    # # for CNN
    # net = neural_network.smallCNN()
    # optimizer = optim.SGD(net.parameters(), lr=0.001)
    # EPOCHS = 3

    # progressbar = ProgressBar(len(orig))

    # for epoch in range(EPOCHS):
    #     print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
    #     print('-' * 22)
    #     # for i, data in enumerate(orig):
    #     # progressbar.progressBar(i)
    #     output = net(orig)
    #     loss = F.mse_loss(output, edit)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     print("Loss: ", loss)

    # torch.save(net.state_dict(), "Model_saved/smallCNN-model.pth")