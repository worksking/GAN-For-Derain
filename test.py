import numpy as np
from common.Densenet import DenseNet
from common.resnet import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim 
import torch.utils.data as data 
from common.utils import *
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
from PIL import ImageEnhance
import cv2
import torchvision.models as models
import torch.backends.cudnn as cudnn


def guided_filter(datas):
    r = 8
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

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    use_cuda = torch.cuda.is_available()
    
    #file = "/data1/szu/szu_rain_4/circle_rain.jpg"
    files = "/home/zgj/projects/Derain/rain_imgs/11.jpg"
    imgs = cv2.imread(files)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)    
    imgs = imgs / 255.0
    Net_G = restore('./saved_nets/net_20180511071209.pkl')
    # Net_G = restore('./saved_nets/best.pkl')
    Net_G.eval()  

    if use_cuda:
        Net_G.cuda()
        Net_G = torch.nn.DataParallel(
            Net_G, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True


    inputs = np.expand_dims(imgs,axis=0)    
    inputs = np.transpose(inputs,(0,3,1,2))
    detail = inputs - guided_filter(inputs)


    inputs = torch.FloatTensor(inputs)
    detail = torch.FloatTensor(detail)
    inputs = Variable(inputs.cuda(), volatile=True)
    detail = Variable(detail.cuda(), volatile=True)

    
    output = Net_G(inputs, detail)
    output = output.data.cpu().numpy()  #Variable 转成numpy
    print(output.shape)
    output = np.squeeze(output,0)   #降维
    output = np.transpose(output,(1,2,0))

    output[np.where(output < 0)] = 0
    output[np.where(output > 1)] = 1  
    #output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB) 

    #cv2.imwrite('./predictions/10.jpg',cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 1)
    plt.imshow(imgs)
    plt.title('input')
    
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    #plt.savefig('./predictions/16.jpg')
    plt.title('output')

    plt.show()
    print("finished test")

    

if __name__ == '__main__':
    main()