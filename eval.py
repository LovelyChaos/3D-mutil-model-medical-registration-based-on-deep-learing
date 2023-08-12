import os
import numpy as np
import math
from PIL import Image
import SimpleITK as sitk
import cv2
from scipy.fftpack import fftshift, ifftshift
from medpy.metric.binary import hd95
from tools import rayleighmode as _rayleighmode
from tools import lowpassfilter as _lowpassfilter
from tools import fft2, ifft2
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from medpy.metric.binary import hd95,hd,assd, asd, dc
import time

start = time.clock()

def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()#打平
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    dsc = (2. * intersection ) / (m1.sum() + m2.sum() + smooth)
    return dsc

def IOU(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()#打平
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    dsc = intersection  / (m1.sum() + m2.sum() - intersection)
    return dsc

def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    '''
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166, 180, 181]'''
    cls_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35]

    #cls_lst = [255]
    dice_lst = []
    for cls in cls_lst:
        #print('cls:  ', cls)
        dice = DSC(pred==cls, gt==cls)
        dice_lst.append(dice)
        #print(dice)
    return np.mean(dice_lst), dice_lst

def compute_label_IOU(gt, pred):
    cls_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
               27, 28, 29, 30, 31, 32, 33, 34, 35]
    #cls_lst = [255]
    IOU_lst = []
    for cls in cls_lst:
        #print('cls:  ', cls)
        dice = IOU(pred==cls, gt==cls)
        IOU_lst.append(dice)
    return np.mean(IOU_lst)

def HD95(pred, target):
    # print("pred.shape: ", pred.shape, "target.shape: ", target.shape)

    return hd95(pred,target)

def compute_label_hd95(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    '''
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166, 180, 181]'''

    cls_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
               27, 28, 29, 30, 31, 32, 33, 34, 35]
    #cls_lst = [255]
    hd95_lst = []
    for cls in cls_lst:
        hd95 = HD95(pred == cls, gt == cls)
        #print('cls:  ', cls ,'hd95:  ', hd95 )
        hd95_lst.append(hd95)
    return np.mean(hd95_lst)
def changedata(image3D):
    numi = -1
    image_empty = np.empty(shape=image3D.shape)#image3D.shape(112, 80, 144)
    for i in range(image3D.shape[0]):
        numi = numi+1
        numj = -1
        img_2D = image3D[i]#img_2D.shape(80, 144)
        for j in range(img_2D.shape[0]):
            numj = numj+1
            numk = -1
            img_1D = img_2D[j]  # img_1D.shape(144)
            for k in range(img_1D.shape[0]):
                numk = numk + 1
                #print(image3D[numi, numj, numk])
                if image3D[numi, numj, numk]!=0:
                    image_empty[numi, numj, numk] = 1
    return image_empty
def compute_mask_dice(gt, pred):
    #gt = changedata(gt)
    #pred = changedata(pred)
    dice = dc(pred, gt)
    dice = np.array(dice)
    return dice

def compute_mask_hd95(gt, pred):
    #gt = changedata(gt)
    #pred = changedata(pred)
    hd95 = HD95(pred, gt)
    return hd95

def compute_mask_asd(gt, pred):
    #gt = changedata(gt)
    #pred = changedata(pred)
    asd1 = asd(pred, gt)
    return asd1

def Get_Ja(displacement):

    D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])


    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])

    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_z[...,0])

    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

    return D1-D2+D3

def NJ_num(img_data):
    Jacb = Get_Ja(img_data)
    image = Jacb[0, :, :, :]
    NJ_num = 0
    NJ_num = np.float32(NJ_num)
    for i in image:
        for j in i:
            for k in j:
                if k < 0:
                    NJ_num = NJ_num + 1
    return NJ_num

def psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0

    psnr = 10 * np.log10(1**2 / mse)
    return psnr

def time_difference(starttime, endtime):
    seconds = ((endtime - starttime).seconds)
    milliseconds = ((endtime - starttime).microseconds)
    difference = (seconds + milliseconds / 1000000)
    return np.float32(difference)

def pcc(img1, img2):
    x = img1.flatten()
    y = img2.flatten()
    n = len(x)

    sum_xy = np.sum(np.sum(x * y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x * x))
    sum_y2 = np.sum(np.sum(y * y))
    pc = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return pc

def mse(img1, img2):
    img1 = img1.flatten()
    img2 = img2.flatten()
    mse = np.sum((img1 - img2) ** 2) / len(img1)
    return mse

def mae(img1, img2):
    img1 = img1.flatten()
    img2 = img2.flatten()
    mae =np.sum(np.absolute(img1 - img2)) / len(img1)
    return mae

def rmse(img1, img2):
    img1 = img1.flatten()
    img2 = img2.flatten()
    mse = np.sum((img1 - img2) ** 2) / len(img1)
    rmse = np.sqrt(mse)
    return rmse

def ssim(y_true, y_pred):
    '''u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 7)
    c2 = np.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)'''
    #return ssim / denom
    a1 = len(y_true.shape)
    b1 = len(y_pred.shape)

    mu1 = y_true.mean()
    mu2 = y_pred.mean()
    sigma1 = np.sqrt(((y_true - mu1) ** 2).mean())
    sigma2 = np.sqrt(((y_pred - mu2) ** 2).mean())
    sigma12 = ((y_true - mu1) * (y_pred - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 1
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12
    return ssim







