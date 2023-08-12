import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import random


class JHUBrainDataset(Dataset):#配对
    def __init__(self, data_path1, data_path2,transforms):
        self.path1 = data_path1
        self.path2 = data_path2
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        range_path = len(self.path1)-1
        #index1 = random.randint(0, range_path)
        #index2 = random.randint(0, range_path)

        #index_pair = np.random.permutation(len(self.paths)) [0:2]
        #print('len(self.path1):',len(self.path1))#50

        #print(index,":", self.path1[index1], self.path2[index2])
        x = pkload(self.path1[index])
        y = pkload(self.path2[index])#Type: <class 'numpy.ndarray'>

        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        #print("x.shape", x.shape)
        #print( "y.shape", y.shape )
        return x, y

    def __len__(self):
        return len(self.path1)

class JHUBrainDataset_nopair(Dataset):#非配对
    def __init__(self, data_path1, data_path2,transforms):
        self.path1 = data_path1
        self.path2 = data_path2
        self.max_num = len(data_path1)-1
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        #index1 = random.randint(0, range_path)
        #index2 = random.randint(0, range_path)

        #index_pair = np.random.permutation(len(self.paths)) [0:2]
        #print('len(self.path1):',len(self.path1))#50

        #print(index,":", self.path1[index], self.path2[index])
        x = pkload(self.path1[index])
        if index!=self.max_num:
            y = pkload(self.path2[index + 1])  # Type: <class 'numpy.ndarray'>
            #print(index, ":", np.count_nonzero(x),np.count_nonzero(y),self.path1[index], self.path2[index + 1])
        else:
            y = pkload(self.path2[0])
            #print(index, ":", np.count_nonzero(x),np.count_nonzero(y),self.path1[index], self.path2[0])

        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        #print("x.shape", x.shape)
        #print( "y.shape", y.shape )
        return x, y

    def __len__(self):
        return len(self.path1)

class JHUBrainDataset_RV(Dataset):#非配对
    def __init__(self, data_path1, data_path2, data_path3, transforms):
        self.path1 = data_path1
        self.path2 = data_path2
        self.path3 = data_path3
        self.max_num = len(data_path1)-1
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        #index1 = random.randint(0, range_path)
        #index2 = random.randint(0, range_path)

        #index_pair = np.random.permutation(len(self.paths)) [0:2]
        #print('len(self.path1):',len(self.path1))#50

        #print(index,":", self.path1[index], self.path2[index])
        x = pkload(self.path1[index])
        if index!=self.max_num:
            y = pkload(self.path2[index + 1])  # Type: <class 'numpy.ndarray'>
            z= pkload(self.path3[index + 1])
            #print(index, ":", np.count_nonzero(x),np.count_nonzero(y),self.path1[index], self.path2[index + 1])
        else:
            y = pkload(self.path2[0])
            z = pkload(self.path3[0])
            #print(index, ":", np.count_nonzero(x),np.count_nonzero(y),self.path1[index], self.path2[0])

        x, y ,z= x[None, ...], y[None, ...], z[None, ...]
        x,y,z = self.transforms([x, y,z])

        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        z = np.ascontiguousarray(z)

        x, y,z = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
        #print("x.shape", x.shape)
        #print( "y.shape", y.shape )
        return x, y, z

    def __len__(self):
        return len(self.path1)


class JHUBrainInferDataset(Dataset):
    def __init__(self, data_path1, data_path2, transforms):
        self.path1 = data_path1
        self.path2 = data_path2
        self.transforms = transforms

    def __getitem__(self, index):
        #print(self.paths)
        path1 = self.path1[index]
        path2 = self.path2[index]
        #print(self.path1[index], self.path2[index])
        #x, x_seg = pkload(path)
        #index_pair = np.random.permutation(len(self.paths)) [0:2]
        #print("self.paths[index index1]：", self.paths[index], self.paths[index1])
        first = pkload(self.path1[index])
        x = first[0]
        x_seg = first[1]
        second = pkload(self.path2[index])
        y = second[0]
        y_seg = second[1]

        x, y = x[None, ...], y[None, ...]
        #x = x[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]

        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x,  x_seg = torch.from_numpy(x),  torch.from_numpy(x_seg)
        y,  y_seg = torch.from_numpy(y),  torch.from_numpy(y_seg)
        return x, x_seg,y,y_seg

    def __len__(self):
        return len(self.path1)
