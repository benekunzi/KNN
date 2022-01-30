import sys
from imagesNetwork import loadMultImages, loadSingleImage
import neural_network
import torch
import torch.nn.functional as F
from progressBar import ProgressBar
import torch.optim as optim
from unsorting import unsorting
import os 
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if not os.path.isfile("Model_saved/model.pth"):
        orig = loadMultImages("Bilder/Bilder original zugeschnitten/")
        edit = loadMultImages("Bilder/Bilder bearbeitet zugeschnitten/")

        o_tensor, e_tensor = unsorting(orig, edit)

        number = o_tensor.size()[0]

        net = neural_network.Net()
        optimizer = optim.SGD(net.parameters(), lr=0.001)
        EPOCHS = 3
        iterations = len(o_tensor[0])
        progessBar = ProgressBar(iterations)

        for epoch in range(EPOCHS):
            print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
            print('-' * 22)
            for j in range(0,number):
                for i in range(0, iterations):
                    progessBar.progressBar(i)
                    for k in range(3):
                        orig_flatten = torch.flatten(o_tensor[j][i][k])
                        edit_flatten = torch.flatten(e_tensor[j][i][k])
                        out = net(orig_flatten)
                        loss = F.mse_loss(out, edit_flatten)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
            print("Loss: ", loss)

        torch.save(net.state_dict(), "Model_saved/model.pth")

    loaded_net = neural_network.Net()
    loaded_net.load_state_dict(torch.load("Model_saved/model.pth"))
    loaded_net.eval()

    large_img = loadSingleImage("Bilder/img_5568-o-test.jpg")

    print("large: ", large_img.size())

    a = torch.zeros(3,3008,4576)
    i = 0
    for h in range(0, 3008, 32):
        for l in range(0, 4576, 32):
            a[:,h:h+32,l:l+32] = large_img[i,:,:,:]
            i += 1

    a *= 255.0
    a = a.int()
    a = a.permute(1,2,0)
    b = a.detach().numpy()
    # plt.figure(1)
    # plt.imshow(a)

    edited_img = torch.zeros(large_img.size())

    with torch.no_grad():
        for i in range(0, len(large_img)):
            for j in range(3):
                flatten = torch.flatten(large_img[i][j])
                out = loaded_net(flatten)
                out = out.unflatten(0, (1,32,32))
                edited_img[i][j] = out

    print("edit", edited_img.size())

    a = torch.zeros(3,3008,4576)
    i = 0
    for h in range(0, 3008, 32):
        for l in range(0, 4576, 32):
            a[:,h:h+32,l:l+32] = edited_img[i,:,:,:]
            i += 1

    a *= 255.0
    a = a.int()
    a = a.permute(1,2,0)
    b = a.detach().numpy()

    plt.figure(2)
    plt.imshow(a)
    plt.show()