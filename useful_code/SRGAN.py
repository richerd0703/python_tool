import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
import math
from torch.autograd import Variable
import time
import datetime
import numpy as np
import os
import cv2
from torchsummary import summary
from PIL import Image
from torchvision import transforms
import random
import glob
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import random
import gdal
import cv2
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


def transform(LR, HR, lr_mean, lr_std, hr_mean, hr_std, lr_min, lr_max, hr_min, hr_max):
    LR = transforms.Resize((target_size // UPSCALE_FACTOR, target_size// UPSCALE_FACTOR))(LR)
    # LR = transforms.Resize((target_size // UPSCALE_FACTOR, target_size// UPSCALE_FACTOR), interpolation=Image.NEAREST)(HR)
    HR = transforms.Resize((target_size, target_size))(HR)
    if random.random() < 0.5:
        LR = transforms.functional.vflip(LR)
        HR = transforms.functional.vflip(HR)
    if random.random() < 0.5:
        LR = transforms.functional.hflip(LR)
        HR = transforms.functional.hflip(HR)
    LR = transforms.ToTensor()(LR).float()
    HR = transforms.ToTensor()(HR).float()
    # LR = (LR - 13000) / 3384
    # HR = (HR - 13000) / 3384
    # LR = (LR / 50 -273.15) / 42.0
    # HR = (HR / 50 - 273.15) / 42.0
    # LR = (LR - lr_min) / (lr_max - lr_min)
    # HR = (HR - hr_min) / (hr_max - hr_min)
    
    # LR = torch.pow(LR, 1.1)
    # HR= torch.pow(HR, 1.1)
    # LR[LR!=LR] = 0 # 设置背景为0
    # HR[HR!=HR] = 0
    # LR = np.array(LR)
    # HR = np.array(HR)
    # hr_numpy = (HR + 273.0) * 50.0 / 65535.0
    # lr_numpy = LR / 65535.0
    # hr_numpy = hr_numpy.reshape(1, 112, 112)
    # lr_numpy = lr_numpy.reshape(1, 7, 7)
    # LR = torch.Tensor(lr_numpy)
    # HR = torch.Tensor(hr_numpy)
    
    # return LR, HR
    
    # return transforms.ToTensor()(LR), transforms.ToTensor()(HR)
    
    # LR = (LR - 0.5) * 2
    # HR = (HR - 0.5) * 2
    # print(LR,HR)
    # LR = (transforms.Normalize(mean = [lr_mean], std = [lr_std])(LR) + 10.0) / 10.0
    # HR = (transforms.Normalize(mean = [hr_mean], std = [hr_std])(HR) + 10.0) / 10.0
    # LR = (LR - 32768) / 32768 #lr_mean#/ lr_std
    # HR = (HR - 32768) / 32768 #hr_mean#/ hr_std
    # LR = transforms.Normalize(mean = [lr_mean], std = [lr_std])(LR)
    # HR = transforms.Normalize(mean = [hr_mean], std = [hr_std])(HR)

    # LR = LR / lr_mean
    # HR = (HR  - hr_mean)/ hr_std * 25.5 + 255.0
    # print(LR, HR)
    
    return LR, HR


class MyDataSet(data.Dataset):
    def __init__(self, path):
        super(MyDataSet, self).__init__()
        self.path = path 
        self.LRpaths = glob.glob(os.path.join(os.path.join(self.path, 'MS'), '*.tif'))
        self.HRpaths = glob.glob(os.path.join(os.path.join(self.path, 'PAN'), '*.tif'))
        if os.path.exists(os.path.join(path, 'normalize.txt')):
            with open(os.path.join(path, 'normalize.txt'), 'r') as f:
                normalize_container = f.read().split('\n')
                self.lr_mean = float(normalize_container[0])
                self.lr_std = float(normalize_container[1])
                self.hr_mean = float(normalize_container[2])
                self.hr_std = float(normalize_container[3])
                self.lr_min = 0#float(normalize_container[4])
                self.lr_max = 0#float(normalize_container[5])
                self.hr_min = 0#float(normalize_container[6])
                self.hr_max = 0#float(normalize_container[7])
        else:
            self.lr_mean = 0
            self.lr_std = 0
            self.hr_mean = 0
            self.hr_std = 0
            self.lr_min = 0
            self.lr_max = 0
            self.hr_min = 0
            self.hr_max = 0
        self.transform = transform

    def __getitem__(self, index):
        lr = Image.open(self.LRpaths[index])
        hr = Image.open(self.HRpaths[index])
       
        # lr, hr = self.transform(lr, hr)
        lr, hr = self.transform(lr, hr, self.lr_mean, self.lr_std, self.hr_mean, self.hr_std, self.lr_min, self.lr_max, self.hr_min, self.hr_max)
        return lr, hr

    def __len__(self):
        return len(self.LRpaths)
    

def Get_DataSet(dataset, length):
    size = len(length)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if size == 1:
        flag = int(length[0] * dataset_size)
        return data.Subset(dataset, indices[:flag])
    elif size == 2:
        flag = int(length[0] * dataset_size)
        return data.Subset(dataset, indices[:flag]), data.Subset(dataset, indices[flag:])
    elif size == 3:
        flag1 = int(length[0] * dataset_size)
        flag2 = int(length[1] * dataset_size)
        return data.Subset(dataset, indices[:flag1]), data.Subset(dataset, indices[flag1:flag2]), data.Subset(dataset, indices[flag2:])

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

# 生成器
class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
    
        self.block1 = nn.Sequential(
            nn.Conv2d(model_input_channel, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, model_input_channel, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(model_input_channel, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

#生成器loss
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True).to(device)
        vgg.features[0]=nn.Conv2d(model_input_channel, 64, kernel_size=3, padding=1)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval().to(device)
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss().to(device)
        self.l1_loss = nn.L1Loss().to(device)
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels).detach()
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)#self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
        # return image_loss + 0.001 * adversarial_loss  + 2e-8 * tv_loss
#loss
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def SSIMnp(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def read_tiff(input_file):
    """
    读取影像
    :param input_file:输入影像
    :return:波段数据，仿射变换参数，投影信息、行数、列数、波段数
    """

    dataset = gdal.Open(input_file)
    rows = dataset.RasterYSize
    cols = dataset.RasterXSize

    geo = dataset.GetGeoTransform()
    proj = dataset.GetProjection()

    couts = dataset.RasterCount

    array_data = np.zeros((couts,rows,cols))

    for i in range(couts):
        band = dataset.GetRasterBand(i+1)
        array_data[i,:,:] = band.ReadAsArray()


    return array_data,geo,proj,rows,cols,couts
    
def write_tiff(output_file,array_data,rows,cols,counts,geo,proj):

    #判断栅格数据的数据类型
    if 'int8' in array_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in array_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    Driver = gdal.GetDriverByName("Gtiff")
    dataset = Driver.Create(output_file,cols,rows,counts,datatype)

    dataset.SetGeoTransform(geo)
    dataset.SetProjection(proj)

    for i in range(counts):
        band = dataset.GetRasterBand(i+1)
        band.WriteArray(array_data[i,:,:])

def hist(src, ref): # 28934724 29256360
    rows, cols = src.shape
    hist1,bins = np.histogram(src.flatten(),256,[0,256]) 
    hist2, bins = np.histogram(ref.flatten(),256,[0,256]) 
    hist1 = hist1.cumsum() / (rows*cols)
    hist2 = hist2.cumsum() / (rows*cols)
    
    # 直方图规定化
    l = np.zeros(256)
    for i in range(256):
        diff = np.abs(hist1[i] - hist2[i])
        matchValue = i
        for j in range(256):
            if np.abs(hist1[i] - hist2[j]) < diff:
                diff = np.abs(hist1[i] - hist2[j])
                matchValue = j
        l[i] = matchValue
    print(l)
    l[0] = 0
    return l[src]

gpu_id = 0
kwargs={'map_location':lambda storage, loc: storage.cuda(gpu_id)}
def load_GPUS(model,model_path,kwargs):
    state_dict = torch.load(model_path,**kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


def train():
    if restore_gmodel_path != '':
        netG.load_state_dict(torch.load(restore_gmodel_path))
        print('netG_model_load_success!')
    if restore_dmodel_path != '':
        netD.load_state_dict(torch.load(restore_dmodel_path))
        print('netD_model_load_success!')

    prev_time = time.time()
    
    for epoch in range(start_epoch, NUM_EPOCHS):

        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0}
        
        # 进入train模式
        netG.train()
        netD.train()
        print('\nTraining---------------------')
        for i, (data, target) in enumerate(train_loader):
            data = data.float()
            target = target.float()
            # print(data, target)
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            real_img = Variable(target).to(device)
            z = Variable(data).to(device)
           
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            
            d_loss =  1 - real_out + fake_out
            
            running_results['d_loss'] += d_loss.item() * batch_size  # d_loss real/fake通过判别器的差距
            d_loss.backward(retain_graph=True)
            # 进行参数优化
            optimizerD.step()
            # schedulerD.step(d_loss)

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            running_results['g_loss'] += g_loss.item() * batch_size
            g_loss.backward()
            optimizerG.step()

            # schedulerG.step(g_loss)

            #-----
            # Log Progress
            #-----
            # Determine approcimate time left
            batches_done = epoch * len(train_loader) + i
            batches_left = NUM_EPOCHS * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()
            p = batches_done / (NUM_EPOCHS * len(train_loader))  * 100
        
            # Print log 
            
            print('\r{0:3f}% [{1}] [Epoch {2:d}/{3:d}] [Batch {4:d}/{5:d}] [D loss: {6:f}] [G loss: {7:f}] ETA: {8}'.format(p, 
            ( ( '#' * int( p / 10 ) ) + ( chr( 48 + int( p % 10 ) ) if p != 100 else '' ) + ( '-' * ( int( ( 100 - p ) / 10 ) - 1 if ( 100 - p ) % 10 == 0 else int( (100 - p) / 10 ) ) ) ), 
            epoch + 1, NUM_EPOCHS, i + 1, len(train_loader),
             running_results['d_loss'] / running_results['batch_sizes'], 
             running_results['g_loss'] / running_results['batch_sizes'], time_left), end = '')

        schedulerG.step()
        schedulerD.step()
        if (epoch+1) % checkpoints_interval == 0 and epoch != 0:
            # save model
            torch.save(netG.state_dict(), final_gmodel_path[:-4] + '_e_{:04d}.pth'.format(epoch+1))
            torch.save(netD.state_dict(), final_dmodel_path[:-4] + '_e_{:04d}.pth'.format(epoch+1))

        if eval_switch:
            eval()    
    # save model
    torch.save(netG.state_dict(), final_gmodel_path)
    torch.save(netD.state_dict(), final_dmodel_path)
    print('model save success!')

        
def eval():
    print('\nEvaling---------------------')
    prev_time = time.time()
    # 进入eval模式 （测试模式参数固定，只有前向传播）
    netG.eval()
    
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    
    with torch.no_grad():

        for i, (val_lr, val_hr) in enumerate(eval_loader):
            batch_size = val_lr.size(0)
            # 已经测试过的数目
            valing_results['batch_sizes'] += batch_size
        
            lr = Variable(val_lr).to(device).float()
            hr = Variable(val_hr).to(device).float()
            

            # 直接输出结果，没有参数优化的过程
            sr = netG(lr)

            lr = lr.to('cpu').detach().numpy()
            hr = hr.to('cpu').detach().numpy()
            sr = sr.to('cpu').detach().numpy()
            
            lr = lr.reshape(batch_size, 1, target_size // UPSCALE_FACTOR, target_size // UPSCALE_FACTOR)
            hr = hr.reshape(batch_size, 1, target_size, target_size)
            sr = sr.reshape(batch_size, 1, target_size, target_size)

            # 计算mse
            batch_mse = ((sr - hr) ** 2).mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = SSIMnp(hr, sr)
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * math.log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

            #-----
            # Log Progress
            #-----
            # Determine approcimate time left
            batches_done = i
            batches_left = len(eval_loader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()
            p = batches_done / len(eval_loader)  * 100
            
            # Print log
            
            print('\r{0:3f}% [{1}] [Batch {2:d}/{3:d}] [converting LR images to SR images PSNR: {4:f} dB SSIM: {5:f}] ETA: {6}'.format(p, 
            ( ( '#' * int( p / 10 ) ) + ( chr( 48 + int( p % 10 ) ) if p != 100 else '' ) + ( '-' * ( int( ( 100 - p ) / 10 ) - 1 if ( 100 - p ) % 10 == 0 else int( (100 - p) / 10 ) ) ) ), 
            i + 1, len(eval_loader),
            valing_results['psnr'], valing_results['ssim'], time_left), end = '')
         

def predict():

    print('\nPredicting---------------------')

    netG.load_state_dict(torch.load(final_gmodel_path))
    
    print('model load success!')

    with open(os.path.join(dataset_path, 'normalize.txt'), 'r') as f:
        normalize_container = f.read().split('\n')
        lr_mean = float(normalize_container[0])
        lr_std = float(normalize_container[1])
        hr_mean = float(normalize_container[2])
        hr_std = float(normalize_container[3])
        # lr_min = float(normalize_container[4])
        # lr_max = float(normalize_container[5])
        # hr_min = float(normalize_container[6])
        # hr_max = float(normalize_container[7])
    predict_paths = glob.glob(os.path.join(os.path.join(dataset_path,'LR'), '*tif'))
    HR_paths = glob.glob(os.path.join(os.path.join(dataset_path,'HR'), '*tif'))
    if restore_switch:
        origin_paths =  glob.glob(os.path.join(os.path.join(origin_dataset_path,'HR'), '*tif'))
    # 进入eval模式 （测试模式参数固定，只有前向传播）
    netG.eval()
    
    
    if not os.path.exists(predict_result_path):
        os.makedirs(predict_result_path)
    if restore_switch:
        if not os.path.exists(predict_restore_result_path):
            os.makedirs(predict_restore_result_path)
    prev_time = time.time()

    with torch.no_grad():

        psnr = 0
        ssim = 0
        if restore_switch:
            restore_psnr = 0
            restore_ssim = 0
        for i, path in enumerate(predict_paths):
            print(path[18:-4])
            value = re.findall(r'[1-9]\d*_[1-9]\d*_[1-9]\d*', path) 

            array_data,geo,proj,rows,cols,couts=read_tiff(path)
            print(geo)
            lr = array_data.reshape(1,1,target_size // UPSCALE_FACTOR,target_size // UPSCALE_FACTOR)
            # lr = torch.Tensor(lr / 65535.0).to(device)
            # lr = torch.Tensor(lr / 255.0).to(device) # new
            
            # lr = (lr - lr_min) / (lr_max - lr_min)
            # lr = np.power(lr, 1.1)
            # lr[lr != lr] = 0 #去除背景值
            lr = torch.Tensor(lr).to(device)
            lr = (lr - 13000)/ 3384.0
            print(lr)
            # 直接输出结果，没有参数优化的过程
            sr = netG(lr)
            
            # sr = sr.to('cpu').detach().numpy() * 65535.0
            # sr = sr.astype(np.int16)
            # sr = sr.to('cpu').detach().numpy() * 255.0
            # sr = sr.astype(np.uint8)
            sr = sr.to('cpu').detach().numpy() *3384.0 +13000#** (10/11)
            print(sr)
            # sr = (sr + hr_min) * (hr_max - hr_min)
            sr = sr.astype(np.uint16)
            sr = sr.reshape(1, target_size, target_size)
            if restore_switch:
                refer_data,refer_geo,refer_proj,refer_rows,refer_cols,refer_couts=read_tiff(origin_paths[i]) 
                sr = sr.reshape(target_size, target_size)
                refer_data = refer_data.reshape(target_size, target_size)
                # 规定化复原
                restore_sr = hist(sr, refer_data)
                sr = sr.reshape(1, target_size, target_size)
                restore_sr = restore_sr.reshape(1, target_size, target_size)


            # 计算mse
            hr,hr_geo,proj,rows,cols,couts=read_tiff(HR_paths[i])
            s_mse = ((sr/255.0 - hr/255.0) ** 2).mean()
            s_ssim = SSIMnp(hr, sr)
            ssim += s_ssim
            s_psnr = 10 * math.log10(1 / (s_mse))
            psnr += s_psnr
            print('[converting LR images to SR images] mse: {:f} PSNR: {:f} dB SSIM: {:f}'.format(s_mse, s_psnr, s_ssim))
            if restore_switch:
                s_restore_mse = ((restore_sr/255.0 - refer_data/255.0) ** 2).mean()
                s_restore_ssim = SSIMnp(refer_data,restore_sr)
                restore_ssim += s_restore_ssim
                s_restore_psnr = 10 * math.log10(1 / (s_restore_mse))
                restore_psnr += s_restore_psnr
                print('[restore] mse:{:f} PSNR: {:f} dB SSIM: {:f}'.format(s_restore_mse, s_restore_psnr, s_restore_ssim))
            b_geo = list(geo)
            b_geo[1] = geo[1] / UPSCALE_FACTOR 
            b_geo[5] = geo[5] / UPSCALE_FACTOR
            save_path = os.path.join(predict_result_path, value[0] + '.tif')
            
            print(sr)
            # save_path = os.path.join(predict_result_path, '{}.tif'.format(i))
            write_tiff(save_path, sr, target_size, target_size, couts, tuple(b_geo), proj) 
            if restore_switch:
                restore_save_path = os.path.join(predict_restore_result_path, value[0] + '.tif')
                write_tiff(restore_save_path, restore_sr, target_size, target_size, couts, tuple(b_geo), proj) 
            #-----
            # Log Progress
            #-----
            # Determine approcimate time left
            batches_done = i
            batches_left = len(predict_paths) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()
            p = batches_done / len(predict_paths)  * 100
            
            # Print log
            
            print('\r{0:3f}% [{1}] [Batch {2:d}/{3:d}] ETA: {4}'.format(p, 
            ( ( '#' * int( p / 10 ) ) + ( chr( 48 + int( p % 10 ) ) if p != 100 else '' ) + ( '-' * ( int( ( 100 - p ) / 10 ) - 1 if ( 100 - p ) % 10 == 0 else int( (100 - p) / 10 ) ) ) ), 
            i + 1, len(predict_paths), time_left), end = '')
        
        psnr_mean = psnr / len(predict_paths)
        ssim_mean = ssim / len(predict_paths)
        print("psnr average:{}, ssim average:{}".format(psnr_mean, ssim_mean))
        if restore_switch:
            restore_psnr_mean = restore_psnr / len(predict_paths)
            restore_ssim_mean = restore_ssim / len(predict_paths)
            print("[restore] psnr average:{}, ssim average:{}".format(restore_psnr_mean, restore_ssim_mean))
        print('\npredict finish.')

def process(img_lr_path, model, pad_num, clip_size, save_main_path, scales):
    array_data,geo,proj,rows,cols,couts = read_tiff(img_lr_path)
    print("lr",rows, cols)
    
    print('sr', rows*scales, cols*scales)
    rows_iters = math.ceil(rows / (clip_size-(pad_num*2)))# + 1 #??
    cols_iters = math.ceil(cols / (clip_size-(pad_num*2)))# + 1 #??
    print(rows / (clip_size-(pad_num*2)), cols / (clip_size-(pad_num*2)))
    print(rows_iters, cols_iters)
    print(array_data.shape)
    
    # this defines that whether this shape fits into the conditon
    l_flag = 0
    if len(array_data.shape) == 2:
        array_data = np.pad(array_data, ((pad_num,pad_num), (pad_num,pad_num)), 'constant', constant_values=0)
    else:
        l_flag = 1
        array_data = np.pad(array_data, ((0,0), (pad_num,pad_num), (pad_num,pad_num)), 'constant', constant_values=0)
    
    if len(array_data.shape) == 2:
        padded_h, padded_w = array_data.shape
    else:
        _, padded_h, padded_w = array_data.shape
    
   

    out = gdal.GetDriverByName("GTiff").Create(save_main_path, cols * scales, rows * scales, couts, gdal.GDT_Byte)

    for i in range(rows_iters): #??
        for j in range(cols_iters): #??
            h_s = i * (clip_size - pad_num * 2)
            h_e = i * (clip_size - pad_num * 2) + clip_size
            w_s = j * (clip_size - pad_num * 2)
            w_e = j * (clip_size - pad_num * 2) + clip_size
            
            r_h = i * (clip_size - pad_num * 2) 
            r_w = j * (clip_size - pad_num * 2) 

            if i == rows_iters - 1:
                h_s = padded_h - clip_size
                h_e = padded_h
                r_h = rows - (clip_size - pad_num * 2) 
            if j == cols_iters - 1:
                w_s = padded_w - clip_size
                w_e = padded_w
                r_w = cols - (clip_size - pad_num * 2) 
            
            print(h_s, h_e, w_s, w_e)
            if l_flag == 1:
                array_data_batch = array_data[:, h_s:h_e, w_s:w_e]
            else:
                array_data_batch = array_data[h_s:h_e, w_s:w_e]
            # X = (array_data_batch -13000)/ 3384.0 #2047.0#255.0
            X = array_data_batch / 255  #2047.0#255.0
            X = X.reshape(1, 3, clip_size, clip_size)
            X = torch.Tensor(X).cuda()
            
            extract_result = model(X)
        
           
            extract_result = extract_result.cpu().detach().numpy() * 255 #2047.0#255.0
            b, c, h, w = extract_result.shape
            extract_result = extract_result.reshape(c, h, w)
            # print(extract_result)
            
            for l in range(c): 
                out.GetRasterBand(l+1).WriteArray(extract_result[l][(pad_num*scales):h-(pad_num*scales), (pad_num*scales):w-(pad_num*scales)], r_w * scales, r_h * scales)
            print(i, j)

    
    geotemp = list(geo)
    geotemp[1] = geotemp[1] / scales #w
    geotemp[5] = geotemp[5] / scales #h
    out.SetGeoTransform(tuple(geotemp)) #写入仿射变换参数
    out.SetProjection(proj) #写入投影
    print('finished.')

def predictWhole():

    print('\nPredicting---------------------')

    netG.load_state_dict(torch.load(final_gmodel_path))
    
    print('model load success!')

    # 进入eval模式 （测试模式参数固定，只有前向传播）
    netG.eval()

    with torch.no_grad():
        process(predict_lr_path, netG, pad_num=4, clip_size=512, save_main_path = out_path, scales=UPSCALE_FACTOR)
        
       

if __name__ == '__main__':

    
    UPSCALE_FACTOR = 4
    start_epoch = 0 #850
    NUM_EPOCHS = 500 #1500
    step_size = 500
    restore_gmodel_path = r''#'./Models/SRGANG_144_e_0230_e_0850.pth' # 不写路径不恢复
    restore_dmodel_path = r''#'./Models/SRGAND_144_e_0850.pth'
    final_gmodel_path = r'G:\dataset\pansharpen\rgb\dataset\train\G_e_0500.pth'    #必须填
    final_dmodel_path = r'G:\dataset\pansharpen\rgb\dataset\train\D.pth'    #必须填
    checkpoints_interval = 20
    single_gpu_batch_size = 2 # 12
    eval_switch = False
    dataset_path = r'G:\dataset\pansharpen\rgb\dataset'##'./2018_dataset_u8_256'

    predict_lr_path = r'G:\dataset\pansharpen\rgb\low_export\25_d3_071.tif'
    out_path = r'G:\dataset\pansharpen\rgb\pre_25_d3_071.tif'

    origin_dataset_path = r'./2018_dataset_u8'
    predict_result_path = './data_144_Results'
    predict_restore_result_path = './112_Restore_Results'
    target_size = 512 #112
    model_input_channel = 3
    restore_switch = False

    netG = Generator(scale_factor = UPSCALE_FACTOR).to(device)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    summary(netG, (3, 7, 7))
    netD = Discriminator().to(device)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    summary(netD, (3, 7, 7))
    
    GPUCount = torch.cuda.device_count()
    if GPUCount > 1:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
    else :
        GPUCount = 1


    #生成器损失函数
    generator_criterion = GeneratorLoss()
    discriminator_criterion = nn.BCELoss().to(device)
   

    optimizerG = torch.optim.Adam(netG.parameters(), lr = 0.0001)
    optimizerD = torch.optim.Adam(netD.parameters(), lr = 0.0001)
  

    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=step_size, gamma=0.5)#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, mode = 'min',factor=0.5, patience=10000, verbose=True)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=step_size, gamma=0.5)
    # 结果集 : loss score psnr（峰值信噪比） ssim（结构相似性）
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    dataset = MyDataSet(dataset_path)
    train_data, eval_data= Get_DataSet(dataset, [0.7, 0.3])

    train_loader = torch.utils.data.DataLoader(dataset=train_data, num_workers=0, batch_size=single_gpu_batch_size * GPUCount, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset=eval_data, num_workers=0, batch_size=single_gpu_batch_size * GPUCount, shuffle=False)
    
    # train()
    # # predict()
    predictWhole()
