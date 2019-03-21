import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dcgan_main import *

if __name__=='__main__':
    

    # training parameters
    batch_size = 128
    lr = 0.0002
    train_epoch = 20

    # data_loader
    img_size = 64
    isCrop = False
    if isCrop:
        transform = transforms.Compose([
            transforms.Scale(108),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    data_dir = 'data/resized_celeba'          # this path depends on your computer
    dset = datasets.ImageFolder(data_dir, transform)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)
    temp = plt.imread(train_loader.dataset.imgs[0][0])
    if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
        sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_preprocess.py\" !!!')
        sys.exit(1)

    # network
    G = generator(128)
    D = discriminator(128)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # results save folder
    if not os.path.isdir('CelebA_DCGAN_results'):
        os.mkdir('CelebA_DCGAN_results')
    if not os.path.isdir('CelebA_DCGAN_results/Random_results'):
        os.mkdir('CelebA_DCGAN_results/Random_results')
    if not os.path.isdir('CelebA_DCGAN_results/Fixed_results'):
        os.mkdir('CelebA_DCGAN_results/Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print('Training start!')
    start_time = time.time()
    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []

        # learning rate decay
        if (epoch+1) == 11:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        if (epoch+1) == 16:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")

        num_iter = 0

        epoch_start_time = time.time()
        for x_, _ in train_loader:
            # train discriminator D
            D.zero_grad()

            if isCrop:
                x_ = x_[:, :, 22:86, 22:86]

            mini_batch = x_.size()[0]

            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)

            x_, y_real_, y_fake_ = x_.cuda(), y_real_.cuda(), y_fake_.cuda()
            D_result = D(x_).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = z_.cuda()
            G_result = G(z_)

            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result.detach().mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.detach().item())

            # train generator G
            G.zero_grad()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = z_.cuda()

            G_result = G(z_)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_)
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data.item())

            num_iter += 1

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time


        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                  torch.mean(torch.FloatTensor(G_losses))))
        p = 'CelebA_DCGAN_results/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
        fixed_p = 'CelebA_DCGAN_results/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
        show_result((epoch+1), save=True, path=p, isFix=False)
        show_result((epoch+1), save=True, path=fixed_p, isFix=True)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save training results")
    torch.save(G.state_dict(), "CelebA_DCGAN_results/generator_param.pkl")
    torch.save(D.state_dict(), "CelebA_DCGAN_results/discriminator_param.pkl")
    with open('CelebA_DCGAN_results/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path='CelebA_DCGAN_results/CelebA_DCGAN_train_hist.png')

    images = []
    for e in range(train_epoch):
        img_name = 'CelebA_DCGAN_results/Fixed_results/CelebA_DCGAN_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave('CelebA_DCGAN_results/generation_animation.gif', images, fps=5)