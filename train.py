import numpy as np
import cv2
from common.Densenet import densenet121

from common.resnet import resnet18
from common.resnet_26 import resnet26
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim 
import torch.utils.data as data 
from common.dataset_test import train_Dataset
from common.dataset_test import val_Dataset

from common.utils import *
import torchvision.models as models
import torch.backends.cudnn as cudnn
from multiprocessing import Process, freeze_support



def guided_filter(datas):
    r = 15
    eps = 1.0
    #batch_size = 1
    batch_size, channel, height, width = datas.shape
    batch_q = np.zeros((batch_size, channel, height, width))
    #print(batch_size,channel, height, width)
    #batch_q = np.zeros((batch_size, channel, height, width))
    batch_q = np.zeros((batch_size, channel, height, width))
    for i in range(int(batch_size)):
        for j in range(int(channel)):
            
            I = datas[i, j, :, :]
            p = datas[i, j, :, :]

            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 *
                                                r + 1), normalize=False, borderType=0)
            mean_I = cv2.boxFilter(
                I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_p = cv2.boxFilter(
                p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_Ip = cv2.boxFilter(
                I * p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(
                I * I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(
                a, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_b = cv2.boxFilter(
                b, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            q = mean_a * I + mean_b
            batch_q[i, j, :, :] = q
            #batch_q[i, :, :, j] = q
    return batch_q


def val_check(Net_D, Net_G, criterion, val_loader=None):
    val_g_error = 0
    Net_D.eval()
    Net_G.eval()
    for t, (x, y) in enumerate(val_loader):
        x_np = x.numpy()
        x_detail = x_np - guided_filter(x_np)
        x_detail = torch.from_numpy(x_detail)
        x_detail = Variable(x_detail.type(torch.cuda.FloatTensor))

        x_train = Variable(x.type(torch.cuda.FloatTensor))
        g_fake_data = Net_G(x_train, x_detail)
        dg_fake_decision = Net_D(g_fake_data)
        val_g_error += criterion(dg_fake_decision, Variable(torch.ones(dg_fake_decision.shape[0],1).type(torch.cuda.FloatTensor))).data[0]
        val_g_error = val_g_error/(t+1)
        if (t + 1) % 1 == 0:
            print('val_g_error: t = %d, loss = %.10f' % (t + 1, val_g_error))    
    return val_g_error





def train(Net_D, Net_G, batch_size, criterion, num_epochs, d_optim, g_optim, d_schedule, g_schedule, loader=None, val_loader=None):
    d_steps = 1
    g_steps = 4
    val_g_error = 0
    best_val_error = 0.01
   
    gi_sampler = torch.FloatTensor(20,3,64,64)
    Net_G.train()
    Net_D.train()
    for epoch in range(num_epochs):                
        for t, (x, y) in enumerate(loader):
            for d_index in range(d_steps):
                Net_D.zero_grad()
                # 1. Train D on real+fake
                x_np = x.numpy()
                x_detail = x_np - guided_filter(x_np)
                x_detail = torch.from_numpy(x_detail)
                x_detail = Variable(x_detail.type(torch.cuda.FloatTensor))
                x_train = Variable(x.type(torch.cuda.FloatTensor))
                y_train = Variable(y.type(torch.cuda.FloatTensor))
                #  1A: Train D on real
                d_real_decision = Net_D(y_train)
                #print(d_real_decision.shape)
                d_real_error = criterion(d_real_decision, Variable(torch.ones(d_real_decision.shape[0], 1).type(torch.cuda.FloatTensor)))  # ones = true
                #d_real_error.backward() # compute/store gradients, but don't change params
                #  1B: Train D on fake               
                d_fake_data = Net_G(x_train, x_detail).detach()  # detach to avoid training G on these labels
                d_fake_decision = Net_D(d_fake_data)
                d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(d_fake_decision.shape[0], 1).type(torch.cuda.FloatTensor)))  # zeros = fake
                #d_fake_error.backward()
                d_loss = d_real_error + d_fake_error
                d_loss.backward()
                d_optim.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
                    
            for g_index in range(g_steps):
                # 2. Train G on D's response (but DO NOT train D on these labels)
                Net_G.zero_grad()
                g_fake_data = Net_G(x_train, x_detail)
                dg_fake_decision = Net_D(g_fake_data)
                g_error = criterion(dg_fake_decision, Variable(torch.ones(dg_fake_decision.shape[0],1).type(torch.cuda.FloatTensor)))  # we want to fool, so pretend it's all genuine
                g_error.backward()
                g_optim.step()  # Only optimizes G's parameters
            if (t + 1) % 1 == 0:
                print('epoch = %d, t = %d, d_loss= %.12f, d_fake_loss=%.12f, g_loss = %.12f' % (epoch + 1, t + 1, d_real_error, d_fake_error, g_error))
        save(Net_G, False, False)
        
        val_g_error = val_check(Net_D, Net_G, criterion, val_loader=val_loader)
        d_schedule.step(val_g_error, epoch=epoch+1)
        g_schedule.step(val_g_error, epoch=epoch+1)

        if val_g_error < best_val_error:
            best_val_error = val_g_error
            print('-------------------------------')
            print('saving net ..............')
            print('-------------------------------')
            save(Net_G, False, True)
            #save(Net_D, False, True)
            print ('finished save')
        # adjust_learning_rate_epoch(g_optim,epoch+1)
        # adjust_learning_rate_epoch(d_optim,epoch+1)

            

def adjust_learning_rate_iteration(optimizer, t , A):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.5 ** (t // A))

def get_generator_input_sampler():
    n,c,h,w =20,3,64,64
    return lambda n,c,h,w: torch.FloatTensor(n,c,h,w)  # Uniform-dist data into generator, _NOT_ Gaussian

def adjust_learning_rate_epoch(optimizer, num_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.1 ** (num_epoch // 1))



def main():    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = torch.cuda.is_available()

    train_datasets = train_Dataset('/data1')
    val_datasets = val_Dataset('/data1')
    batch_size = 100
    train_loader = data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=10)

    Net_D = densenet121()
    #print(Net_D)
    Net_G = resnet26()    
    if use_cuda:
        Net_D.cuda()
        Net_G.cuda()
        Net_D = torch.nn.DataParallel(Net_D, device_ids=range(torch.cuda.device_count()))
        Net_G = torch.nn.DataParallel(Net_G, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
 
    #data params
    d_learning_rate = 0.5  # 2e-4
    g_learning_rate = 0.5
    optim_betas = (0.9, 0.999)
    num_epochs = 100
        
    criterion = nn.BCEWithLogitsLoss() 
    #criterion = nn.MSELoss() 
    #d_optim = optim.Adam(Net_D.parameters(), lr=d_learning_rate, betas=optim_betas)
    #g_optim = optim.Adam(Net_G.parameters(), lr=g_learning_rate, betas=optim_betas)
    d_optim = optim.SGD(params=Net_D.parameters(), lr=d_learning_rate, weight_decay=1e-10)
    g_optim = optim.SGD(params=Net_G.parameters(), lr=g_learning_rate, weight_decay=1e-10)
    #d_optim = optim.RMSprop(Net_D.parameters(), lr=0.001, weight_decay=1e-10, centered=False)
    #g_optim = optim.RMSprop(Net_G.parameters(), lr=0.001, weight_decay=1e-10, centered=False)


    d_schedule = optim.lr_scheduler.ReduceLROnPlateau(d_optim, mode='min', factor=0.1, verbose=True, patience=2)
    g_schedule = optim.lr_scheduler.ReduceLROnPlateau(g_optim, mode='min', factor=0.1, verbose=True, patience=1)

    
    train(Net_D, Net_G, batch_size, criterion,  num_epochs, d_optim, g_optim, d_schedule, g_schedule, loader=train_loader, val_loader=val_loader)

if __name__ == '__main__':
    main()
    
        
