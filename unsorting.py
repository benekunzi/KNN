import torch
import random

def unsorting(ot, et):
    o_tensor = ot
    e_tensor = et
    orig = torch.zeros(ot.size())
    edit = torch.zeros(et.size())
    n = ot.size()[0]
    m = ot.size()[1]
    liste = list(range(0,m*n))

    for i in range(len(liste)):
        rand = random.randint(0, len(liste)-1)
        rand = liste.pop(rand)
        orig[int(rand/m),rand%m,:,:,:] = o_tensor[int(i/m),i%m,:,:,:]
        edit[int(rand/m),rand%m,:,:,:] = e_tensor[int(i/m),i%m,:,:,:]

    return (orig, edit)