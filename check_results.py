import torch
import neural_network
from downsize import loadSingleImage
import cv2
import numpy as np
import matplotlib.pyplot as plt


choice = input("Choose network\n0: MLP\n1: CNN\n")

if choice == "0":
    grayscale = True
    loaded_net = neural_network.Net()
    loaded_net.load_state_dict(torch.load("Model_saved/model.pth"))
    loaded_net.eval()

    batch = loadSingleImage("Bilder/img_5568-o-test.jpg", grayscale=grayscale)
    output = torch.zeros(batch.size())

    with torch.no_grad():
        if grayscale:
            flat = torch.flatten(batch)
            out = loaded_net(flat)
            output = out.unflatten(0, (752,1144))
        else:
            for i in range(3):
                flat = torch.flatten(batch[i,:,:])
                out = loaded_net(flat)
                output[i] = out.unflatten(0, (1, 1144, 752))

    output *= 255.0
    output = output.int()
    if not grayscale:
        output = output.permute(1,2,0)
    img = output.detach().numpy()
    fin = np.array(img, dtype=np.uint8)
    print(fin)

    plt.imshow(fin)
    plt.show()

    # cv2.imshow('Final', img)
    # cv2.waitKey(0)

elif choice == "1":
    loaded_net = neural_network.smallCNN()
    loaded_net.load_state_dict(torch.load("Model_saved/smallCNN-model.pth"))
    loaded_net.eval()

    batch = loadSingleImage("Bilder/img_5568-o-test.jpg")
    print(batch.size())

    output = torch.zeros(batch.size())
    print(output.size())

    with torch.no_grad():
        output = loaded_net(batch.unsqueeze(0))

    output *= 255.0
    output = output.int()
    output = torch.squeeze(output, 0)
    print(output.size())
    output = output.permute(1, 2 ,0)
    img = output.detach().numpy()
    fin = np.array(img, dtype=np.uint8)

    cv2.imshow('Final', fin)
    cv2.waitKey(0)
