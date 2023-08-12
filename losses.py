import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from math import exp
import math


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        #dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            #dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy)# + torch.mean(dz)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross

def ncc_loss(I, J, win=None):
    '''
    输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    '''
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(0))
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3DiTV(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self):
        super(Grad3DiTV, self).__init__()
        a = 1

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, :-1, 1:, 1:])
        dx = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, :-1, 1:])
        dz = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, 1:, :-1])
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
        d = torch.mean(torch.sqrt(dx+dy+dz+1e-6))
        grad = d / 3.0
        return grad

def Get_Ja(displacement):

    '''

    Calculate the Jacobian value at each point of the displacement map having

    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3

    '''

    D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])



    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])

    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_z[...,0])

    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

    return D1-D2+D3

class NJ_loss(torch.nn.Module):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''

    def __init__(self):
        super( NJ_loss, self ).__init__( )

    def forward(self, y_pred, y_true):
        Neg_Jac = 0.5 * (torch.abs( Get_Ja( y_pred ) ) - Get_Ja( y_pred ))
        print(Get_Ja( y_pred ).shape, torch.abs( Get_Ja( y_pred ) ), Get_Ja( y_pred ))
        print("torch.sum(Neg_Jac)", torch.sum(Neg_Jac))
        return torch.sum(Neg_Jac)

class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC, self).__init__()
        self.win = win

    def forward(self, y_pred, y_true):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    
class MutualInfomation(torch.nn.Module):
    """
    Soft Mutual Information approximation for intensity volumes and probabilistic volumes
    (e.g. probabilistic segmentaitons)
    More information/citation:
    - Courtney K Guo.
      Multi-modal image registration with unsupervised deep learning.
      PhD thesis, Massachusetts Institute of Technology, 2019.
    - M Hoffmann, B Billot, JE Iglesias, B Fischl, AV Dalca.
      Learning image registration without images.
      arXiv preprint arXiv:2004.10282, 2020. https://arxiv.org/abs/2004.10282
    - Modified by Junyu Chen, original Keras code: https://github.com/adalca/neurite/blob/dev/neurite/tf/metrics.py
    Includes functions that can compute mutual information between volumes,
      between segmentations, or between a volume and a segmentation map
    mi = MutualInformation()
    mi.volumes
    mi.segs
    mi.volume_seg
    mi.channelwise
    mi.maps
    """
    def __init__(self, type='volumes', bin_centers=None, nb_bins=None, min_clip=None, max_clip=None, soft_bin_alpha=1):
        super(MutualInfomation, self).__init__()
        """
        Initialize the mutual information class
        Arguments below are related to soft quantizing of volumes, which is done automatically 
        in functions that comptue MI over volumes (e.g. volumes(), volume_seg(), channelwise()) 
        using these parameters
        Args:
           bin_centers (np.float32, optional): array or list of bin centers. 
               Defaults to None.
           nb_bins (int, optional):  number of bins, if bin_centers is not specified. 
               Defaults to 16.
           min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
           max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
           soft_bin_alpha (int, optional): alpha in RBF of soft quantization. Defaults to 1.
        """
        self.type = type
        self.bin_centers = None
        if bin_centers is not None:
            self.bin_centers = torch.from_numpy(bin_centers).cuda().float()
            assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]

        self.nb_bins = nb_bins
        if bin_centers is None and nb_bins is None:
            self.nb_bins = 16

        self.min_clip = min_clip
        if self.min_clip is None:
            self.min_clip = -np.inf

        self.max_clip = max_clip
        if self.max_clip is None:
            self.max_clip = np.inf

        self.soft_bin_alpha = soft_bin_alpha

    def volumes(self, x, y):
        """
        Mutual information for each item in a batch of volumes.
        Algorithm:
        - use neurite.utils.soft_quantize() to create a soft quantization (binning) of
          intensities in each channel
        - channelwise()
        Parameters:
            x and y:  [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = x.shape[1]
        tensor_channels_y = y.shape[1]
        msg = 'volume_mi requires two single-channel volumes. See channelwise().'
        assert tensor_channels_x == 1, msg
        assert tensor_channels_y == 1, msg

        # volume mi
        return torch.flatten(self.channelwise(x, y))

    def segs(self, x, y):
        """
        Mutual information between two probabilistic segmentation maps.
        Wraps maps()
        Parameters:
            x and y:  [bs, nb_labels, ...]
        Returns:
            Tensor of size [bs]
        """
        # volume mi
        return self.maps(x, y)

    def volume_seg(self, x, y):
        """
        Mutual information between a volume and a probabilistic segmentation maps.
        Wraps maps()
        Parameters:
            x and y: a volume and a probabilistic (soft) segmentation. Either:
              - x: [bs, ..., 1] and y: [bs, ..., nb_labels], Or:
              - x: [bs, ..., nb_labels] and y: [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = x.shape[1]
        tensor_channels_y = y.shape[1]
        msg = 'volume_seg_mi requires one single-channel volume.'
        assert min(tensor_channels_x, tensor_channels_y) == 1, msg
        msg = 'volume_seg_mi requires one multi-channel segmentation.'
        assert max(tensor_channels_x, tensor_channels_y) > 1, msg

        # transform volume to soft-quantized volume
        if tensor_channels_x == 1:
            x = self._soft_sim_map(x[:, 0, ...])  # [bs, B, ...]
        else:
            y = self._soft_sim_map(y[:, 0, ...])  # [bs, B, ...]

        return self.maps(x, y)  # [bs]

    def channelwise(self, x, y):
        """
        Mutual information for each channel in x and y. Thus for each item and channel this
        returns retuns MI(x[...,i], x[...,i]). To do this, we use neurite.utils.soft_quantize() to
        create a soft quantization (binning) of the intensities in each channel
        Parameters:
            x and y:  [bs, ..., C]
        Returns:
            Tensor of size [bs, C]
        """
        # check shapes
        tensor_shape_x = x.shape
        tensor_shape_y = y.shape
        assert tensor_shape_x == tensor_shape_y, 'volume shapes do not match'
        # reshape to [bs, V, C]
        if len(tensor_shape_x) != 3:
            x = torch.reshape(x, (tensor_shape_x[0], tensor_shape_x[1], -1))  # [bs, C, V]
            x = x.permute(0, 2, 1)# [bs, V, C]
            y = torch.reshape(y, (tensor_shape_x[0], tensor_shape_x[1], -1))  # [bs, C, V]
            y = y.permute(0, 2, 1)  # [bs, V, C]

        # move channels to first dimension
        cx = x.permute(2, 0, 1) # [C, bs, V]
        cy = y.permute(2, 0, 1) # [C, bs, V]

        # soft quantize
        cxq = self._soft_sim_map(cx)  # [C, bs, V, B]
        cyq = self._soft_sim_map(cy)  # [C, bs, V, B]
        # get mi
        cout = []
        for i in range(cxq.shape[0]):
            cout.append(self.maps(cxq[i:i+1, ...], cyq[i:i+1, ...]))
        cout = torch.stack(cout, dim=0) # [C, bs]

        # permute back
        return cout.permute(1, 0) # [bs, C]

    def maps(self, x, y):
        """
        Computes mutual information for each entry in batch, assuming each item contains
        probability or similarity maps *at each voxel*. These could be e.g. from a softmax output
        (e.g. when performing segmentaiton) or from soft_quantization of intensity image.
        Note: the MI is computed separate for each itemin the batch, so the joint probabilities
        might be  different across inputs. In some cases, computing MI actoss the whole batch
        might be desireable (TODO).
        Parameters:
            x and y are probability maps of size [bs, ..., B], where B is the size of the
              discrete probability domain grid (e.g. bins/labels). B can be different for x and y.
        Returns:
            Tensor of size [bs]
        """

        # check shapes
        tensor_shape_x = x.shape
        tensor_shape_y = y.shape
        assert tensor_shape_x == tensor_shape_y, 'volume shapes do not match'
        assert torch.min(x) >= 0, 'voxel values must be non-negative'
        assert torch.min(y) >= 0, 'voxel values must be non-negative'

        eps = 1e-6

        # reshape to [bs, V, B]
        if len(tensor_shape_x) != 3:
            x = torch.reshape(x, (tensor_shape_x[1], tensor_shape_x[2], tensor_shape_x[3])) # [bs, V, B1]
            y = torch.reshape(y, (tensor_shape_x[1], tensor_shape_x[2], tensor_shape_x[3])) # [bs, V, B2]

        # x probability for each batch entry
        px = torch.sum(x, 1, keepdim=True)  # [bs, 1, B1]
        px = px / torch.sum(px, dim=2, keepdim=True)
        # y probability for each batch entry
        py = torch.sum(y, 1, keepdim=True)  # [bs, 1, B2]
        py = py / torch.mean(py, dim=2, keepdim=True)

        # joint probability for each batch entry
        x_trans = x.permute(0, 2, 1)  # [bs, B1, V]
        pxy = torch.bmm(x_trans, y)  # [bs, B1, B2]
        pxy = pxy / (torch.sum(pxy, dim=[1, 2], keepdim=True) + eps)  # [bs, B1, B2]

        # independent xy probability
        px_trans = px.permute(0, 2, 1)  # [bs, B1, 1]
        pxpy = torch.bmm(px_trans, py)  # [bs, B1, B2]
        pxpy_eps = pxpy + eps

        # mutual information
        log_term = torch.log(pxy / pxpy_eps + eps)  # [bs, B1, B2]
        mi = torch.sum(pxy * log_term, dim=[1, 2])  # [bs]
        return mi

    def _soft_log_sim_map(self, x):
        """
        soft quantization of intensities (values) in a given volume
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image.
        Returns:
            volume with one more dimension [bs, ..., B]
        """

        return self.soft_quantize(x,
                                  alpha=self.soft_bin_alpha,
                                  bin_centers=self.bin_centers,
                                  nb_bins=self.nb_bins,
                                  min_clip=self.min_clip,
                                  max_clip=self.max_clip,
                                  return_log=True)  # [bs, ..., B]

    def _soft_sim_map(self, x):
        """
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image.
        Returns:
            volume with one more dimension [bs, ..., B]
        """
        return self.soft_quantize(x,
                                  alpha=self.soft_bin_alpha,
                                  bin_centers=self.bin_centers,
                                  nb_bins=self.nb_bins,
                                  min_clip=self.min_clip,
                                  max_clip=self.max_clip,
                                  return_log=False)  # [bs, ..., B]

    def _soft_prob_map(self, x, **kwargs):
        """
        normalize a soft_quantized volume at each voxel, so that each voxel now holds a prob. map
        Parameters:
            x [bs, ..., B]: soft quantized volume
        Returns:
            x [bs, ..., B]: renormalized so that each voxel adds to 1 across last dimension
        """
        eps = 1e-6
        x_hist = self._soft_sim_map(x, **kwargs)  # [bs, ..., B]
        x_hist_sum = torch.sum(x_hist, -1, keepdim=True) + eps  # [bs, ..., B]
        x_prob = x_hist / x_hist_sum  # [bs, ..., B]
        return x_prob

    def soft_quantize(self, x,
                      bin_centers=None,
                      nb_bins=16,
                      alpha=1,
                      min_clip=-np.inf,
                      max_clip=np.inf,
                      return_log=False):
        """
        (Softly) quantize intensities (values) in a given volume, based on RBFs.
        In numpy this (hard quantization) is called "digitize".

        Code modified based on:
        https://github.com/adalca/neurite/blob/3858b473fcdc89354fe645a453d75ad01c794c8a/neurite/tf/utils/utils.py#L860
        """
        if bin_centers is not None:
            if not torch.is_tensor(bin_centers):
                bin_centers = torch.from_numpy(bin_centers).cuda().float()
            else:
                bin_centers = bin_centers.cuda().float()
            #assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]
        else:
            if nb_bins is None:
                nb_bins = 16
            # get bin centers dynamically
            minval = torch.min(x)
            maxval = torch.max(x)
            bin_centers = torch.linspace(minval.item(), maxval.item(), nb_bins)
        #print(bin_centers)

        # clipping at bin values
        x = x[..., None]  # [..., 1]
        x = torch.clamp(x, min_clip, max_clip)

        # reshape bin centers to be (1, 1, .., B)
        new_shape = [1] * (len(x.shape) - 1) + [nb_bins]
        bin_centers = torch.reshape(bin_centers, new_shape)  # [1, 1, ..., B]

        # compute image terms
        bin_diff = torch.square(x - bin_centers.cuda())  # [..., B]
        log = -alpha * bin_diff  # [..., B]

        if return_log:
            return log  # [..., B]
        else:
            return torch.exp(log)  # [..., B]

    def forward(self, y_pred, y_true):
        if self.type.lower() == 'volumes':
            mi = self.volumes(y_pred, y_true)
        elif self.type.lower() == 'segmentation':
            mi = self.segs(y_pred, y_true)
        elif self.type.lower() == 'volume segmentation':
            mi = self.volume_seg(y_pred, y_true)
        elif self.type.lower() == 'channelwise':
            mi = self.channelwise(y_pred, y_true)
        else:
            raise Exception("Type not implemented!")
        return -mi.mean()


def thirdOrderSplineKernel(u):
    abs_u = u.abs()
    sqr_u = abs_u.pow(2.0)

    result = torch.FloatTensor(u.size()).zero_()
    result = result.to(u.device)

    mask1 = abs_u < 1.0
    mask2 = (abs_u >= 1.0) & (abs_u < 2.0)

    result[mask1] = (4.0 - 6.0 * sqr_u[mask1] + 3.0 * sqr_u[mask1] * abs_u[mask1]) / 6.0
    result[mask2] = (8.0 - 12.0 * abs_u[mask2] + 6.0 * sqr_u[mask2] - sqr_u[mask2] * abs_u[mask2]) / 6.0

    return result


class MILoss(nn.Module):
    def __init__(self, num_bins=16):
        super(MILoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, moving, fixed):
        moving = moving.view(moving.size(0), -1)
        fixed = fixed.view(fixed.size(0), -1)

        padding = float(2)
        batchsize = moving.size(0)

        fixedMin = fixed.min(1)[0].view(batchsize, -1)
        fixedMax = fixed.max(1)[0].view(batchsize, -1)

        movingMin = moving.min(1)[0].view(batchsize, -1)
        movingMax = moving.max(1)[0].view(batchsize, -1)
        # print(fixedMax,movingMax)

        JointPDF = torch.FloatTensor(batchsize, self.num_bins, self.num_bins).zero_()
        movingPDF = torch.FloatTensor(batchsize, self.num_bins).zero_()
        fixedPDF = torch.FloatTensor(batchsize, self.num_bins).zero_()
        JointPDFSum = torch.FloatTensor(batchsize).zero_()
        JointPDF_norm = torch.FloatTensor(batchsize, self.num_bins, self.num_bins).zero_()

        if JointPDF.device != moving.device:
            JointPDF = JointPDF.to(moving.device)
            movingPDF = movingPDF.to(moving.device)
            fixedPDF = fixedPDF.to(moving.device)
            JointPDFSum = JointPDFSum.to(moving.device)
            JointPDF_norm = JointPDF_norm.to(moving.device)

        # print(JointPDF.device)

        fixedBinSize = (fixedMax - fixedMin) / float((self.num_bins - 2 * padding))
        movingBinSize = (movingMax - movingMin) / float(self.num_bins - 2 * padding)

        fixedNormalizeMin = fixedMin / fixedBinSize - float(padding)
        movingNormalizeMin = movingMin / movingBinSize - float(padding)

        # print(fixed.shape,fixedBinSize.shape,fixedNormalizeMin.shape)
        fixed_winTerm = fixed / fixedBinSize - fixedNormalizeMin

        fixed_winIndex = fixed_winTerm.int()
        fixed_winIndex[fixed_winIndex < 2] = 2
        fixed_winIndex[fixed_winIndex > (self.num_bins - 3)] = self.num_bins - 3

        moving_winTerm = moving / movingBinSize - movingNormalizeMin

        moving_winIndex = moving_winTerm.int()
        moving_winIndex[moving_winIndex < 2] = 2
        moving_winIndex[moving_winIndex > (self.num_bins - 3)] = self.num_bins - 3

        for b in range(batchsize):
            a_1_index = moving_winIndex[b] - 1
            a_2_index = moving_winIndex[b]
            a_3_index = moving_winIndex[b] + 1
            a_4_index = moving_winIndex[b] + 2

            a_1 = thirdOrderSplineKernel((a_1_index - moving_winTerm[b]))
            a_2 = thirdOrderSplineKernel((a_2_index - moving_winTerm[b]))
            a_3 = thirdOrderSplineKernel((a_3_index - moving_winTerm[b]))
            a_4 = thirdOrderSplineKernel((a_4_index - moving_winTerm[b]))
            for i in range(self.num_bins):
                fixed_mask = (fixed_winIndex[b] == i)
                fixedPDF[b][i] = fixed_mask.sum()
                for j in range(self.num_bins):
                    JointPDF[b][i][j] = a_1[fixed_mask & (a_1_index == j)].sum() + a_2[
                        fixed_mask & (a_2_index == j)].sum() + a_3[fixed_mask & (a_3_index == j)].sum() + a_4[
                                            fixed_mask & (a_4_index == j)].sum()

            # JointPDFSum[b] = JointPDF[b].sum()
            # norm_facor = 1.0 / JointPDFSum[b]
            # print(JointPDF[b])
            JointPDF_norm[b] = JointPDF[b] / JointPDF[b].sum()
            fixedPDF[b] = fixedPDF[b] / fixed.size(1)

        movingPDF = JointPDF_norm.sum(1)

        # print(JointPDF_norm)

        MI_loss = torch.FloatTensor(batchsize).zero_().to(moving.device)
        for b in range(batchsize):
            JointPDF_mask = JointPDF_norm[b] > 0
            movingPDF_mask = movingPDF[b] > 0
            fixedPDF_mask = fixedPDF[b] > 0

            MI_loss[b] = (JointPDF_norm[b][JointPDF_mask] * JointPDF_norm[b][JointPDF_mask].log()).sum() \
                         - (movingPDF[b][movingPDF_mask] * movingPDF[b][movingPDF_mask].log()).sum() \
                         - (fixedPDF[b][fixedPDF_mask] * fixedPDF[b][fixedPDF_mask].log()).sum()

        # print(MI_loss)
        loss = MI_loss.mean()
        return -1.0 * loss