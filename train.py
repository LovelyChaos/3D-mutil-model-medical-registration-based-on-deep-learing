import os, utils, glob, losses
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.system('pip install TensorboardX')
os.system('pip install ml_collections')
os.system('pip install natsort')
#os.system('pip install nibabel')
os.system('pip install timm')
os.system('pip install fairscale')
os.system('pip install einops')
#os.system('pip install -U --pre statsmodels')
#os.system('pip install antspyx')
os.system('pip install medpy')
#os.system('pip install SimpleITK')
#import ants
import random
import torch, models
from torch.utils.data import DataLoader
from data import datasets, trans
from eval import *
import numpy as np

from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from models import CONFIGS as CONFIGS_ViT_seg
from natsort import natsorted
from losses import NJ_loss
from Net import *
import datetime
import TransMorph
from rmi3D import RMI3DLoss
import configs_TransMorph as configs
import CrossFormer
import Biformer
import davit
CONFIGS = {
    'TransMorph': configs.get_3DTransMorph_config(),
    'Davit': configs.get_3D_Davit(),
    'BiFormer': configs.get_3D_BiFormer(),
    'CrossFormer': configs.get_3D_CrossFormer(),
}

''''''
import moxing as mox

mox.file.copy_parallel('obs://aaa11177/mutil_modality/data/', 'aaa11177/mutil_modality/data/')
mox.file.copy_parallel('obs://aaa11177/mutil_modality/datasets/T1_train/', 'T1_train/')
mox.file.copy_parallel('obs://aaa11177/mutil_modality/datasets/T2_train/', 'T2_train/')
mox.file.copy_parallel('obs://aaa11177/mutil_modality/datasets/T1_train1/', 'T1_train1/')
mox.file.copy_parallel('obs://aaa11177/mutil_modality/datasets/T2_train1/', 'T2_train1/')
mox.file.copy_parallel('obs://aaa11177/mutil_modality/datasets/T1_test/', 'T1_test/')
mox.file.copy_parallel('obs://aaa11177/mutil_modality/datasets/T2_test/', 'T2_test/')
mox.file.copy_parallel('obs://aaa11177/mutil_modality/checkpoint/', 'checkpoint/')  ###读
mox.file.copy_parallel('obs://aaa11177/mutil_modality/logtxt/', 'logtxt/')  ###读
mox.file.copy_parallel('obs://aaa11177/mutil_modality/result/', 'result/')  ###读
mox.file.copy_parallel('obs://aaa11177/mutil_modality/nii/', 'nii/')  ###读
mox.file.copy_parallel('obs://aaa11177/mutil_modality/val_pictures/', 'val_pictures/')  ###读


def save_nii(img, name):
    def_out2 = img.squeeze(0)
    def_out2 = def_out2.squeeze(0)
    new_data = def_out2
    nii_img = nb.load('nii/aligned_norm-281.nii.gz')
    affine = nii_img.affine.copy()
    hdr = nii_img.header.copy()
    new_nii = nb.Nifti1Image(new_data, affine, hdr)
    # 保存nii文件，后面的参数是保存的文件名
    nb.save(new_nii, name)
    # nb.save( new_nii, save_dir + '/' + str( zhi ) + 'test_x.nii' )
    mox.file.copy_parallel('result/', 'obs://aaa11177/mutil_modality/result/')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)


def main():
    batch_size = 1
    imagesize = (160, 160, 160)
    if_reduce_variance = False
    train_T1_dir = 'T1_train/'
    train_T2_dir = 'T2_train/'
    val_T1_dir = 'T1_test/'
    val_T2_dir = 'T2_test/'
    save_dir = 'result/'
    log_file = 'logtxt/'
    log_file_name = 'MI_BiFormer.txt'
    lr = 0.0001
    max_epoch = 101
    spatial_trans = SpatialTransformer(imagesize, mode='bilinear')

    config_vit = CONFIGS_ViT_seg['ViT-V-Net']

    #model = models.ViTVNet(config_vit, img_size = (160, 160, 160))
    #model = models.vm2(img_size=(160, 160, 160))
    #model = CopyX2_downsampleMaxPool(img_size = imagesize)
    #config = CONFIGS['TransMorph']
    #model = TransMorph.TransMorph(config)
    #model = TransMorph_double_decoder.TransMorph(config)
    #config = CONFIGS['Davit']
    #model = davit.Davit(config)
    config = CONFIGS['BiFormer']
    model = Biformer.BiFormer_Unet(config)
    #config = CONFIGS['CrossFormer']
    #model = CrossFormer.CrossFormer_MFB(config)


    exists_txt = os.path.exists(log_file + log_file_name)

    '''----------loss-----------'''
    # criterion = nn.MSELoss()
    criterion = losses.MILoss()
    #criterion = losses.NCC(win = [9,9,9])
    #criterion = RMI3DLoss(with_logits = False, radius = 5, stride = 5, padding = 0)
    criterions = [criterion]
    # prepare deformation loss
    criterions += [losses.Grad3d(penalty = 'l2')]
    weights = [1]
    weights += [0.02]

    if exists_txt == True:
        data = np.loadtxt(log_file + log_file_name, dtype=np.float32)
        start_epoch = len(data) + 1
        print('start_epoch', start_epoch)
        epoch_start = start_epoch
        model_dir = 'checkpoint/'
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        model_lists = natsorted(glob.glob(model_dir + '*'))
        model_lists = model_lists[::-1]
        # print( "model_lists:",model_lists )
        last_model = torch.load(model_lists[0])['state_dict']
        print("last_model file name:", model_lists[0])
        model.load_state_dict(last_model)
    else:
        updated_lr = lr
        epoch_start = 1


    model.cuda()
    # '''原始dataloader
    train_composed = transforms.Compose([  # trans.RandomFlip(0),
        trans.NumpyType((np.float32, np.float32)),
    ])

    val_composed = transforms.Compose([  # trans.Seg_norm(), #rearrange segmentation label to 1 to 35
        trans.NumpyType((np.float32, np.int16)),
    ])

    path1 = natsorted(glob.glob(train_T1_dir + '*.pkl'))
    path2 = natsorted(glob.glob(train_T2_dir + '*.pkl'))


    train_set = datasets.JHUBrainDataset_nopair(path1, path2, transforms=train_composed)
    val_set = datasets.JHUBrainDataset_nopair(natsorted(glob.glob(val_T1_dir + '*.pkl')),
                                            natsorted(glob.glob(val_T2_dir + '*.pkl')), transforms=train_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
    print("len(train_loader):", len(train_loader))
    print("len(val_loader):", len(val_loader))

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)

    best_mse = 0
    for epoch in range(epoch_start, 2*max_epoch):
        f1 = open(os.path.join(log_file, log_file_name), "a+")
        mox.file.copy_parallel('logtxt/', 'obs://aaa11177/mutil_modality/logtxt/')
        print('*****Training Starts*****')
        print("Epoch:  ", epoch)
        '''
        Training
        '''
        loss_all = AverageMeter()
        idx = 0
        for x, y in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            # data = [t.cuda() for t in data]
            x = x.cuda().float()
            y = y.cuda().float()
            x = x.squeeze(5)
            y = y.squeeze(5)
            if if_reduce_variance==True:
                y2x=fixed2moving(y, imagesize = imagesize)
                y2x=y2x.reshape(shape = (1, 1, 160, 160, 160))
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                y2x = y2x.to(device)
                y2x = y2x.float()
                x_in = torch.cat((x, y2x), dim = 1)
            else:
                x_in = torch.cat((x, y), dim = 1)
            output = model(x_in)
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss

            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output
            #flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in)
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                'Iter {} of {} loss {:.6f}, Img Sim: {:.6f}, Reg: {:.9f}'.format(idx, len(train_loader),
                                                                                 loss.item(),
                                                                                 loss_vals[0].item() / 2,
                                                                                 loss_vals[1].item() / 2, ))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        eval_asd= AverageMeter()
        i = 0
        print('*****Val Starts*****')
        with torch.no_grad():
            for x, y in val_loader:
                i = i + 1
                model.eval()
                x = x.cuda().float()
                #x_seg = x_seg.cuda().float()
                y = y.cuda().float()
                #y_seg = y_seg.cuda().float()
                x = x.squeeze(5)
                y = y.squeeze(5)
                #x_seg = x_seg.squeeze(5)
                #y_seg = y_seg.squeeze(5)


                x_in = torch.cat((x, y), dim = 1)
                output = model(x_in)
                warped = output[0]
                warped = warped.cpu().numpy().squeeze()

                y=y.cpu().numpy().squeeze()

                '''保存部分验证后的图像，方便可视化'''
                if i <5:
                    x = x.cpu().numpy().squeeze()
                    save_pic_dir = 'val_pictures/'
                    save_pic(x, warped, y, i, save_pic_dir)



                dsc = compute_mask_dice(warped, y)
                asd = compute_mask_asd(warped, y)
                print(i ,"dsc:", dsc, "asd:", asd)
                eval_asd.update(asd.item())
                eval_dsc.update(dsc.item())
        print("eval_asd.avg: ", eval_asd.avg)
        print("eval_dsc.avg: ", eval_dsc.avg)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, save_dir='checkpoint/', filename='epoch{}asd{:.6f}dsc{:.6f}.pth.tar'.format(str(epoch),  eval_asd.avg, eval_dsc.avg))
        mox.file.copy_parallel('checkpoint/', 'obs://aaa11177/mutil_modality/checkpoint/')
        f1.write(str(loss_all.avg)  + " " + str(eval_asd.avg)+ " " + str(eval_dsc.avg))
        mox.file.copy_parallel('logtxt/', 'obs://aaa11177/mutil_modality/logtxt/')
        f1.write("\r")  # 换行
        mox.file.copy_parallel('logtxt/', 'obs://aaa11177/mutil_modality/logtxt/')
        f1.close()
        loss_all.reset()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='modes', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir + filename)
    mox.file.copy_parallel('checkpoint/', 'obs://aaa11177/mutil_modality/checkpoint/')
    '''删除模型，华为云上没用
    model_lists = natsorted(glob.glob(save_dir + '*'))
    mox.file.copy_parallel('checkpoint/', 'obs://aaa11177/mutil_modality/checkpoint/')
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        mox.file.copy_parallel('checkpoint/', 'obs://aaa11177/mutil_modality/checkpoint/')
        model_lists = natsorted(glob.glob(save_dir + '*'))
        mox.file.copy_parallel('checkpoint/', 'obs://aaa11177/mutil_modality/checkpoint/')'''

def save_pic(fixed, moving, warped, i, save_dir):
    plt.figure()
    channel = fixed.shape[0] # 0是冠状面，1是横断面，2是矢状面
    for s in range(channel):

        if (s%10==0 and s>59 and s<101):
            slicer_fixed = fixed[s, :, :]  # 0是冠状面，1是横断面，2是矢状面
            slicer_moving = moving[s, :, :]
            slicer_warped = warped[s, :, :]
            plt.subplot(131)
            #current_axes = plt.subplot(131)#这3行是去掉坐标轴
            #current_axes.get_xaxis().set_visible(False)
            #current_axes.get_yaxis().set_visible(False)
            plt.imshow(slicer_fixed, cmap = 'gray')

            plt.subplot(132)
            #current_axes = plt.subplot(132)
            #current_axes.get_xaxis().set_visible(False)
            #current_axes.get_yaxis().set_visible(False)
            plt.imshow(slicer_moving, cmap = 'gray')

            plt.subplot(133)
            #current_axes = plt.subplot(133)
            #current_axes.get_xaxis().set_visible(False)
            #current_axes.get_yaxis().set_visible(False)
            plt.imshow(slicer_warped, cmap = 'gray')
            filename = str(i) + '/'
            if not os.path.exists(save_dir + filename):
                os.mkdir(save_dir + filename)

            plt.savefig(save_dir + filename+ str(s) + '.png')#savefig()一定要在show()之前调用！！！
            mox.file.copy_parallel('val_pictures/', 'obs://aaa11177/mutil_modality/val_pictures/')
            plt.show()
    plt.close()
def fixed2moving(x, imagesize):
    x = x.cpu().numpy()
    x = np.reshape(x, imagesize)
    r = np.ones(shape = imagesize)  # image3D.shape(112, 80, 144)
    for i in range(0, imagesize[0]):
        for j in range(0, imagesize[1]):
            for k in range(0, imagesize[2]):
                if x[i, j, k] != 0:
                    r[i, j, k] = x[i, j, k]
    return torch.from_numpy(r)
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
    '''GPU configuration'''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
