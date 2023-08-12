import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np

class Merging_Convdown(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_channel = 48, kernel_size = 3, stride=1):
        super().__init__()
        self.dim = dim
        self.increase_channal = self.Conv_block(self.dim * 8, out_c = out_channel, kernel = 3, stride=1,)

    def Conv_block(self, in_c, out_c, kernel, stride):
        layer = nn.Sequential(
            nn.Conv3d(in_c, out_channels = out_c, kernel_size = kernel, stride = stride, padding = 1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(out_c),
        )
        return layer

    def forward(self, x):
        """
        x: B, H*W*T, C
        """
        B, C, H, W, T  = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, T % 2))

        x0 = x[:, :, 0::2, 0::2, 0::2]  # B C H/2 W/2 T/2
        x1 = x[:, :, 1::2, 0::2, 0::2]  # B C H/2 W/2 T/2
        x2 = x[:, :, 0::2, 1::2, 0::2]  # B C H/2 W/2 T/2
        x3 = x[:, :, 0::2, 0::2, 1::2]  # B C H/2 W/2 T/2
        x4 = x[:, :, 1::2, 1::2, 0::2]  # B C H/2 W/2 T/2
        x5 = x[:, :, 0::2, 1::2, 1::2]  # B C H/2 W/2 T/2
        x6 = x[:, :, 1::2, 0::2, 1::2]  # B C H/2 W/2 T/2
        x7 = x[:, :, 1::2, 1::2, 1::2]  # B C H/2 W/2 T/2

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim = 1)  # B 8*C H/2 W/2 T/2
        x = self.increase_channal(x) # B out_channel H/2 W/2 T/2

        return x

class CopyX2_Merging_Convdown(nn.Module):  #Total params: 355,875
    ''' 输入图像复制一份 '''
    def __init__(self,  img_size = (160, 160, 160)):
        super(CopyX2_Merging_Convdown, self).__init__()
        self.sub_net = copyX2_Merging_Convdown(img_size=img_size)
        self.spatial_trans = SpatialTransformer( img_size )

    def forward(self, x):
        moving = x[:, 0:1, :, :]  # [B, 1, H, W, D]
        #fixed = x[:, 1:2, :, :] # [B, 1, H, W, D]
        flow = self.sub_net(x)
        warped = self.spatial_trans( moving, flow )
        return warped , flow

class copyX2_Merging_Convdown(nn.Module):
    def __init__(self, enc_nf = [16, 32, 64, 128], dec_nf = [128, 64, 32, 32, 32, 16, 16], img_size=(64, 256, 256), mode='bilinear'):
        super(copyX2_Merging_Convdown, self).__init__()

        self.enc = nn.ModuleList()
        self.enc.append(Merging_Convdown(4, enc_nf[0], 3,1))
        self.enc.append(Merging_Convdown(enc_nf[0], enc_nf[1], 3,1))
        self.enc.append(Merging_Convdown(enc_nf[1], enc_nf[2], 3,1))
        self.enc.append(Merging_Convdown(enc_nf[2], enc_nf[3], 3,1))
        self.enc.append(Merging_Convdown(enc_nf[3], enc_nf[3], 3,1))

        # Decoder
        self.dec = nn.ModuleList()
        self.dec.append(self.decoder_block(enc_nf[-1], dec_nf[0], 3))  # 1
        self.dec.append(self.decoder_block(dec_nf[0] * 2, dec_nf[1], 3))  # 2
        self.dec.append(self.decoder_block(dec_nf[1] * 2, dec_nf[2], 3))  # 3
        self.dec.append(self.decoder_block(dec_nf[2] * 2, dec_nf[3], 3))  # 4
        self.dec.append(self.decoder_block(dec_nf[3] + enc_nf[0], dec_nf[4], 3))  # 5

        self.restore1 = self.decoder_block(dec_nf[4], dec_nf[5], 3)
        self.restore2 = self.decoder_block(dec_nf[5], dec_nf[6], 3)
        self.flow = nn.Conv3d(dec_nf[-1], 3, kernel_size = 3, padding = 1)
        self.downsample = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        moving = x[:, 0:1, :, :]  # [B, 1, H, W, D]
        fixed = x[:, 1:2, :, :] # [B, 1, H, W, D]
        # encoder
        moving = torch.cat([moving, moving], dim = 1)
        fixed = torch.cat([fixed, fixed], dim = 1)

        x = torch.cat([moving, fixed], dim = 1)
        skip_conv = []

        x = self.enc[0](x)
        skip_conv.append(x)

        #x = self.downsample(x)
        x = self.enc[1](x)
        skip_conv.append(x)

        #x = self.downsample(x)
        x = self.enc[2](x)
        skip_conv.append(x)

        #x = self.downsample(x)
        x = self.enc[3](x)
        skip_conv.append(x)

        #x = self.downsample(x)
        x = self.enc[4](x)

        # decoder + skip conv

        x = self.dec[0](x)
        x = self.upsample(x)

        x = torch.cat([x, skip_conv[3]], dim = 1)
        x = self.dec[1](x)
        x = self.upsample(x)

        x = torch.cat([x, skip_conv[2]], dim = 1)
        x = self.dec[2](x)
        x = self.upsample(x)

        x = torch.cat([x, skip_conv[1]], dim = 1)
        x = self.dec[3](x)
        x = self.upsample(x)

        x = torch.cat([x, skip_conv[0]], dim = 1)
        x = self.dec[4](x)
        x = self.upsample(x)

        # Upsample to full res, concatenate and conv
        x = self.restore1(x)
        x = self.restore2(x)
        flow = self.flow(x)


        return flow

    def decoder_block(self, in_c, out_c, kernel):

        layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size = kernel, stride = 1, padding = 1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(out_c),
        )

        return layer


class CopyX2(nn.Module):  #Total params: 355,875
    ''' 输入图像复制一份 '''
    def __init__(self,  img_size = (160, 160, 160)):
        super(CopyX2, self).__init__()
        self.sub_net = copyX2(img_size=img_size)
        self.spatial_trans = SpatialTransformer( img_size )

    def forward(self, x):
        moving = x[:, 0:1, :, :]  # [B, 1, H, W, D]
        #fixed = x[:, 1:2, :, :] # [B, 1, H, W, D]
        flow = self.sub_net(x)
        warped = self.spatial_trans( moving, flow )
        return warped , flow

class copyX2(nn.Module):
    def __init__(self, enc_nf = [16, 32, 32, 32], dec_nf = [32, 32, 32, 32, 32, 16, 16], img_size=(64, 256, 256), mode='bilinear'):
        super(copyX2, self).__init__()

        self.enc = nn.ModuleList()
        self.enc.append(self.encoder_block(4, enc_nf[0], 3,1))
        self.enc.append(self.encoder_block(enc_nf[0], enc_nf[1], 3,2))
        self.enc.append(self.encoder_block(enc_nf[1], enc_nf[2], 3,2))
        self.enc.append(self.encoder_block(enc_nf[2], enc_nf[3], 3,2))
        self.enc.append(self.encoder_block(enc_nf[3], enc_nf[3], 3,2))

        # Decoder
        self.dec = nn.ModuleList()
        self.dec.append(self.decoder_block(enc_nf[-1], dec_nf[0], 3))  # 1
        self.dec.append(self.decoder_block(dec_nf[0] * 2, dec_nf[1], 3))  # 2
        self.dec.append(self.decoder_block(dec_nf[1] * 2, dec_nf[2], 3))  # 3
        self.dec.append(self.decoder_block(dec_nf[2] * 2, dec_nf[3], 3))  # 4
        self.dec.append(self.decoder_block(dec_nf[3] + enc_nf[0], dec_nf[4], 3))  # 5

        self.restore1 = self.decoder_block(dec_nf[4], dec_nf[5], 3)
        self.restore2 = self.decoder_block(dec_nf[5], dec_nf[6], 3)
        self.flow = nn.Conv3d(dec_nf[-1], 3, kernel_size = 3, padding = 1)
        self.downsample = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        moving = x[:, 0:1, :, :]  # [B, 1, H, W, D]
        fixed = x[:, 1:2, :, :] # [B, 1, H, W, D]
        # encoder
        moving = torch.cat([moving, moving], dim = 1)
        fixed = torch.cat([fixed, fixed], dim = 1)

        x = torch.cat([moving, fixed], dim = 1)
        skip_conv = []

        x = self.enc[0](x)
        skip_conv.append(x)

        #x = self.downsample(x)
        x = self.enc[1](x)
        skip_conv.append(x)

        #x = self.downsample(x)
        x = self.enc[2](x)
        skip_conv.append(x)

        #x = self.downsample(x)
        x = self.enc[3](x)
        skip_conv.append(x)

        #x = self.downsample(x)
        x = self.enc[4](x)

        # decoder + skip conv

        x = self.dec[0](x)
        x = self.upsample(x)

        x = torch.cat([x, skip_conv[3]], dim = 1)
        x = self.dec[1](x)
        x = self.upsample(x)

        x = torch.cat([x, skip_conv[2]], dim = 1)
        x = self.dec[2](x)
        x = self.upsample(x)

        x = torch.cat([x, skip_conv[1]], dim = 1)
        x = self.dec[3](x)
        x = self.upsample(x)

        x = torch.cat([x, skip_conv[0]], dim = 1)
        x = self.dec[4](x)

        # Upsample to full res, concatenate and conv
        x = self.restore1(x)
        x = self.restore2(x)
        flow = self.flow(x)


        return flow
    def encoder_block(self, in_c, out_c, kernel, stride):

        layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size = kernel, stride = stride, padding = 1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(out_c),
        )

        return layer

    def decoder_block(self, in_c, out_c, kernel):

        layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size = kernel, stride = 1, padding = 1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(out_c),
        )

        return layer

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        #print("self.grid.shape", self.grid.shape)
        #print( "flow.shape", flow.shape )
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

if __name__ == '__main__':
    from torchinfo import summary
    model = copyX2_Merging_Convdown()
    #model = CopyX2()
    summary(model, (1,2, 160, 160, 160), depth=3)

