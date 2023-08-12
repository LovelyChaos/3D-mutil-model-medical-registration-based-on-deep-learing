import torch
import numpy as np
import torch.nn.functional as F
import unfoldNd
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn










if __name__ == '__main__':
    up_Transpose = nn.ConvTranspose3d(in_channels = 96, out_channels = 96, kernel_size = 3,
                                               stride = 2, padding = 1, output_padding = 1, bias = False)
    batch_size, classes, height, width, deep = 1, 96, 5, 5, 5
    x = torch.randn(batch_size, classes, height, width, deep, requires_grad = False)
    y = torch.randn(batch_size, classes, height, width, deep, requires_grad = False)
    print(x.shape)
    x = up_Transpose(x)
    print(x.shape)
