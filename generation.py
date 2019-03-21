import torch
import matplotlib.pyplot as plt
from dcgan_main import *

params = torch.load('CelebA_DCGAN_results/generator_param.pkl')
G=generator()
G.load_state_dict(params)

def gen_image(z):
    img=G(z)
    img=(img.cpu().detach().numpy().squeeze().transpose(1,2,0) +1) /2.0
    plt.imshow(img)
    
def rand_image():
    z=torch.randn((1, 100)).view(-1, 100, 1, 1)
    gen_image(z)

#linear interpolation of z

z1=torch.randn((1, 100)).view(-1, 100, 1, 1)
fig=plt.figure(figsize=(8, 8))
columns = 6
rows = 6
for i in range(1, columns*rows +1):
    temp=z1.clone()
    temp[0][30]=(i-17)
    img=G(temp)
    img=(img.cpu().detach().numpy().squeeze().transpose(1,2,0) +1) /2.0
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

