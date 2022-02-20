import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        size_input = 752*1144
        self.fc1 = nn.Linear(size_input, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, size_input)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # encode
        # size of 3008x4576 with 1 channel (rgb layers pass individually)
        self.conv1 = nn.Conv2d(1, 2, (int(3008/128), int(4576/128))) # input channels, depth, kernel size
        # output size = ((w-f)/s + 1, (w-f)/s + 1) = ((3008-128)/1 + 1), (4576-128)/1 + 1) = (2986, 4542)
        # after one maxpool layer it becomes (2, 1493, 2271)
        self.conv2 = nn.Conv2d(2, 4, (int(1493/64), int(2271/64)))
        # output size = ((w-f)/s + 1, (w-f)/s + 1) = ((1493-128)/1 + 1), (2271-128)/1 + 1) = (1366, 2144)
        # after one maxpool layer it becomes (2, 1493, 2271)
        self.conv3 = nn.Conv2d(4, 8, (int(735/32), int(1118/32)))
        # output size = ((w-f)/s + 1, (w-f)/s + 1) = ((1493-128)/1 + 1), (2271-128)/1 + 1) = (1366, 2144)
        # after one maxpool layer it becomes (2, 1493, 2271)     
        self.conv4 = nn.Conv2d(8, 16, (int(357/16), int(542/16)))   

        self.pool = nn.MaxPool2d(2,2)

        # size of input: 16*168*255
        self.fc1 = nn.Linear(16*168*255, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 16*336*510)

        # decode
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=(int(357/16), int(542/16)), stride=2)
        self.up2 = nn.ConvTranspose2d(16, 8, kernel_size=(int(735/32), int(1118/32)), stride=2)
        self.up21 = nn.ConvTranspose2d(8, 8, kernel_size=(26, 34))
        self.up3 = nn.ConvTranspose2d(8, 4, kernel_size=(int(1493/64), int(2271/64)), stride=2)
        self.up4 = nn.ConvTranspose2d(4, 2, kernel_size=(int(3008/128), int(4576/128)), stride=2)
        self.up5 = nn.ConvTranspose2d(2, 1, kernel_size=(3008, 4576))
        self.dec1 = nn.Conv2d(32, 16, 3)
        self.dec2 = nn.Conv2d(16, 8, 3)
        self.dec3 = nn.Conv2d(8, 4, 3)
        self.dec4 = nn.Conv2d(4, 2, 3)
        self.dec5 = nn.Conv2d(2, 1, 3)

        self.final = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1, padding=0))

    def concat(self, x1, x2):
        if x1.shape == x2.shape:
            return torch.cat((x1, x2), 1)
        else:
            print(x1.shape, x2.shape)
            raise ValueError('concatenation failed: wrong dimensions')

    def forward(self, x):
        e1 = F.relu(self.conv1(x))
        print("size after conv1: ", e1.size())
        m1 = self.pool(e1)
        print("size after conv1 pool: ", m1.size())
        e2 = F.relu(self.conv2(m1))
        print("size after conv2: ", e2.size())
        m2 = self.pool(e2)
        print("size after conv2 pool: ", m2.size())
        e3 = F.relu(self.conv3(m2))
        print("size after conv3: ", e3.size())
        m3 = self.pool(e3)
        print("size after conv3 pool: ", m3.size())
        e4 = F.relu(self.conv4(m3))
        print("size after conv4: ", e4.size())
        m4 = self.pool(e4)
        print("size after conv4 pool: ", m4.size())

        x = m4.view(m4.size(0), -1)
        print("size after flatten: ", x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        print("size after fully connected layers: ", x.size())

        x = x.unflatten(-1, (16, 336, 510))
        print("size after unflatten: ", x.size())

        c1 = self.concat(x, e4)                                     # shape of x and e4: 16, 336, 510
        print("size of c1: ", c1.size())                     # shape: 32, 336, 510
        d1 = self.dec1(c1)
        print("size of d1: ", d1.size())                      # shape: 16, 334, 508

        u2 = self.up2(d1)                                           # shape: 8, 714, 1085
        print("size of u2: ", u2.size())
        u21 = self.up21(u2)
        print("size of u21: ", u21.size())

        c2 = self.concat(u2, e3)                                    # shape: 16, 714, 1085
        print("size of c2: ", c2.size())

        d2 = self.dec2(c2)                                          # shape: 8, 712, 1083
        print("size of d2: ", d2.size())

        u3 = self.up3(d2)                                           # shape: 4, 
        print("size of u3: ", u3.size())

        c3 = self.concat(u3, e2)
        print("size of c3: ", c3.size())

        d3 = self.dec3(c3)
        print("size of d3: ", d3.size())

        u4 = self.up4(d3) 
        print("size of u4: ", u4.size())

        c4 = self.concat(u4, e1)
        print("size of c4: ", c4.size())

        d4 = self.dec4(c4)
        print("size of d4: ", d4.size())

class smallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode1 = nn.Conv2d(3, 6, 3)
        self.encode11 = nn.Conv2d(6,6,3)
        self.encode2 = nn.Conv2d(6, 12, 3)
        self.encode21 = nn.Conv2d(12,12,3)
        self.encode3 = nn.Conv2d(12, 24, 3)
        self.encode31 = nn.Conv2d(24,24,3)
        self.encode4 = nn.Conv2d(24, 48, 3)
        self.encode41 = nn.Conv2d(48,48,3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mid1 = nn.Conv2d(48, 96, 3)
        self.mid2 = nn.Sequential( 
            nn.Conv2d(kernel_size=3, in_channels=96, out_channels=96, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(0.5))

        self.up1 = nn.ConvTranspose2d(96, 48, kernel_size= (11, 10), stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(48, 24, kernel_size= 5, stride=2)
        self.up21 = nn.ConvTranspose2d(24, 24, kernel_size= 6)
        self.up3 = nn.ConvTranspose2d(24, 12, kernel_size= 5, stride=2)
        self.up31 = nn.ConvTranspose2d(12, 12, kernel_size= 7)
        self.up4 = nn.ConvTranspose2d(12, 6, kernel_size= 3, stride=2)
        self.up41 = nn.ConvTranspose2d(6, 6, kernel_size= 8)
        self.up5 = nn.ConvTranspose2d(6, 3, kernel_size= 5)
        self.dec1 = nn.Conv2d(96, 48, 3)
        self.dec2 = nn.Conv2d(48, 24, 3)
        self.dec3 = nn.Conv2d(24, 12, 3)
        self.dec4 = nn.Conv2d(12, 6, 3)
        self.dec5 = nn.Conv2d(6, 3, 7)
        self.final = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1, padding=0))

    def concat(self, x1, x2):
        if x1.shape == x2.shape:
            return torch.cat((x1, x2), 1)
        else:
            print(x1.shape, x2.shape)
            raise ValueError('concatenation failed: wrong dimensions')

    def forward(self, x):
        e1 = F.relu(self.encode1(x))
        e11 = F.relu(self.encode11(e1))
        m1 = self.pool(e11)
        e2 = F.relu(self.encode2(m1))
        e21 = F.relu(self.encode21(e2))
        m2 = self.pool(e21)
        e3 = F.relu(self.encode3(m2))
        e31 = F.relu(self.encode31(e3))
        m3 = self.pool(e31)
        e4 = F.relu(self.encode4(m3))
        e41 = F.relu(self.encode41(e4))
        m4 = self.pool(e41)

        midle1 = self.mid1(m4)
        midle2 = self.mid2(midle1)

        u1 = self.up1(midle2)
        c1 = self.concat(u1, e4)         
        d1 = self.dec1(c1)
        u2 = self.up2(d1)  
        u21 = self.up21(u2)
        c2 = self.concat(u21, e3)
        d2 = self.dec2(c2)                                  
        u3 = self.up3(d2)                                        
        u31 = self.up31(u3)
        c3 = self.concat(u31, e2)
        d3 = self.dec3(c3)
        u4 = self.up4(d3) 
        u41 = self.up41(u4)
        c4 = self.concat(u41, e1)
        d4 = self.dec4(c4)
        u5 = self.up5(d4)

        logits = self.final(u5)
        return torch.sigmoid(logits)
# class smallCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encode1 = nn.Conv2d(1, 2, 3)
#         self.encode2 = nn.Conv2d(2, 4, 3)
#         self.encode3 = nn.Conv2d(4, 8, 3)
#         self.encode4 = nn.Conv2d(8, 16, 3)

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.fc1 = nn.Linear(49680, 1000)
#         self.fc2 = nn.Linear(1000, 2000)
#         self.fc3 = nn.Linear(2000, 200160)

#         self.up1 = nn.ConvTranspose2d(32, 16, kernel_size= 5, stride=2)
#         self.up2 = nn.ConvTranspose2d(16, 8, kernel_size= 5, stride=2)
#         self.up21 = nn.ConvTranspose2d(8, 8, kernel_size= 6)
#         self.up3 = nn.ConvTranspose2d(8, 4, kernel_size= 5, stride=2)
#         self.up31 = nn.ConvTranspose2d(4, 4, kernel_size= 7)
#         self.up4 = nn.ConvTranspose2d(4, 2, kernel_size= 3, stride=2)
#         self.up41 = nn.ConvTranspose2d(2, 2, kernel_size= 8)
#         self.up5 = nn.ConvTranspose2d(2, 1, kernel_size= 5)
#         self.dec1 = nn.Conv2d(32, 16, 3)
#         self.dec2 = nn.Conv2d(16, 8, 3)
#         self.dec3 = nn.Conv2d(8, 4, 3)
#         self.dec4 = nn.Conv2d(4, 2, 3)
#         self.dec5 = nn.Conv2d(1, 1, 7)
#         self.final = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1, padding=0))

#     def concat(self, x1, x2):
#         if x1.shape == x2.shape:
#             return torch.cat((x1, x2), 1)
#         else:
#             print(x1.shape, x2.shape)
#             raise ValueError('concatenation failed: wrong dimensions')

#     def forward(self, x):
#         e1 = F.relu(self.encode1(x))
#         m1 = self.pool(e1)
#         e2 = F.relu(self.encode2(m1))
#         m2 = self.pool(e2)
#         e3 = F.relu(self.encode3(m2))
#         m3 = self.pool(e3)
#         e4 = F.relu(self.encode4(m3))
#         m4 = self.pool(e4)

#         x = m4.view(m4.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = x.unflatten(-1, (16, 139, 90))

#         c1 = self.concat(x, e4)         
#         d1 = self.dec1(c1)
#         u2 = self.up2(d1)  
#         u21 = self.up21(u2)
#         c2 = self.concat(u21, e3)
#         d2 = self.dec2(c2)                                  
#         u3 = self.up3(d2)                                        
#         u31 = self.up31(u3)
#         c3 = self.concat(u31, e2)
#         d3 = self.dec3(c3)
#         u4 = self.up4(d3) 
#         u41 = self.up41(u4)
#         c4 = self.concat(u41, e1)
#         d4 = self.dec4(c4)
#         u5 = self.up5(d4)

#         logits = self.final(u5)
#         return torch.sigmoid(logits)