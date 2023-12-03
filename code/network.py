
from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, relu
import torchvision.models as models
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import logging
from torch.distributions.uniform import Uniform

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        # print(x.shape) # torch.Size([24, 1, 256, 256])
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    ################################################   added the output before self.out_conv ###############################
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        feature_map = [x4]
        x = self.up1(x4, x3)
        feature_map.append(x)
        x = self.up2(x, x2)
        feature_map.append(x)
        x = self.up3(x, x1)
        feature_map.append(x)
        x = self.up4(x, x0)
        feature_map.append(x)
        output = self.out_conv(x)
        return output, feature_map


class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class Decoder_URPC(nn.Module):
    def __init__(self, params):
        super(Decoder_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chns, class_num, train_encoder=True, train_decoder=True):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.train_encoder = train_encoder
        self.train_decoder = train_decoder

        if not (train_encoder):
            for params in self.encoder.parameters():
                params.requires_grad = False
                params = params.detach_()

        if not (train_decoder):
            for params in self.decoder.parameters():
                params.requires_grad = False
                params = params.detach_()

    def forward(self, x):
        feature = self.encoder(x)
        output, _ = self.decoder(feature)
        return output, feature[-1]

class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)
        self.aux_decoder3 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)
        aux3_feature = [FeatureDropout(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3


class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_URPC(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(
            feature, shape)
        return dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg


class UNet_DS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DS, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_DS(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv1 = bn_relu_conv2d(in_channels, out_channels, kernel)
        self.conv2 = bn_relu_conv2d(in_channels, out_channels, kernel)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

class bn_relu_conv2d(nn.Module):
    def __init__(self, c_i, c_o, k, drop=0.05):
        super(bn_relu_conv2d, self).__init__()
        self.batch_norm = nn.BatchNorm2d(c_i)
        self.conv = nn.Conv2d(in_channels=c_i, out_channels=c_o, kernel_size=k, padding='same')
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.batch_norm(x)
        x = relu(x)
        x = self.drop(self.conv(x))
        return x

class bn_relu_deconv2d(nn.Module):
    def __init__(self, c_i, c_o, k, drop=0.05):
        super(bn_relu_deconv2d, self).__init__()
        self.batch_norm = nn.BatchNorm2d(c_i)
        self.conv = nn.ConvTranspose2d(in_channels=c_i, out_channels=c_o, kernel_size=k, stride=2)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.batch_norm(x)
        x = relu(x)
        x = self.drop(self.conv(x))
        return x


class encoder(nn.Module):
    def __init__(self, feature_base=32, drop=0.05):
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=feature_base, kernel_size=3, padding='same') # keep_prob: 1
        self.drop1 = nn.Dropout(drop)
        self.res1 = ResidualBlock(in_channels=feature_base, out_channels=feature_base, kernel=3)

        self.conv2 = nn.Conv2d(in_channels = feature_base, out_channels= 2* feature_base, kernel_size=3, padding='same')
        self.drop2 = nn.Dropout(drop)
        self.res2 = ResidualBlock(in_channels=2*feature_base, out_channels=2*feature_base, kernel=3)


        self.conv3 = nn.Conv2d(in_channels= 2 * feature_base, out_channels= 4 * feature_base, kernel_size=3, padding='same')
        self.drop3 = nn.Dropout(drop)
        self.res3 = ResidualBlock(in_channels=4 * feature_base, out_channels=4 * feature_base, kernel=3)

        self.conv4 = nn.Conv2d(in_channels= 4 * feature_base, out_channels=8 * feature_base, kernel_size=3, padding='same')
        self.drop4 = nn.Dropout(drop)
        self.res4 = ResidualBlock(in_channels=8 * feature_base, out_channels=8 * feature_base, kernel=3)

        self.conv5 = nn.Conv2d(in_channels=8 * feature_base, out_channels=16 * feature_base, kernel_size=3, padding='same')
        self.drop5 = nn.Dropout(drop)
        self.res5 = ResidualBlock(in_channels=16 * feature_base, out_channels=16 * feature_base, kernel=3)

        self.res5_2 = ResidualBlock(in_channels=16 * feature_base, out_channels=16 * feature_base, kernel=3)
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        conv1 = self.drop1(self.conv1(x))
        res1 = self.res1(conv1)
        res1_pool = self.max_pool(res1)

        conv2 = self.drop2(self.conv2(res1_pool))
        res2 = self.res2(conv2)
        res2_pool = self.max_pool(res2)

        conv3 = self.drop3(self.conv3(res2_pool))
        res3 = self.res3(conv3)
        res3_pool = self.max_pool(res3)

        conv4 = self.drop4(self.conv4(res3_pool))
        res4 = self.res4(conv4)
        res4_pool = self.max_pool(res4)

        conv5 = self.drop5(self.conv5(res4_pool))
        res5 = self.res5(conv5)
        res5_2 = self.res5_2(res5)

        return (res1, res2, res3, res4, res5_2)


class res_encoder(nn.Module):
    def __init__(self,x):
        super(res_encoder, self).__init__()
        if x == '18':
            self.model = models.resnet18(pretrained=True)
        if x=='34':
            self.model = models.resnet34(pretrained=True)
        if x=='50':
            # print('x')
            self.model = models.resnet50(pretrained=True)
    def forward(self, x):
        conv1 = self.model.conv1(x) # [5,64,192,192]
        bnorm1 = self.model.bn1(conv1)
        relu1 = self.model.relu(bnorm1)
        pool1 = self.model.maxpool(relu1) # [5,64,96,96]
        res1 = self.model.layer1(pool1) # [5,64,96,96]
        res2 = self.model.layer2(res1) # [5,128/512,48,48]
        res3 = self.model.layer3(res2) # [5,256/1024,24,24]
        res4 = self.model.layer4(res3) # [5,512/2048,12,12]
        return (conv1,res1,res2,res3,res4)


class ViT_encoder(nn.Module):
    def __init__(self):
        super(ViT_encoder, self).__init__()
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        self.model = ViT_seg(config_vit, img_size=256)
        data = np.load('/home/chenyu/Documents/PointRendMedical/code/IL_medical_segmentation-main/R50+ViT-B_16.npz')
        self.model.load_from(data)
    def forward(self, x):
        transformer = self.model.transformer(x)
        encoded = transformer[0]
        B, n_patch, hidden = encoded.shape
        h, w = int(np.sqrt(n_patch)),int(np.sqrt(n_patch))
        encoded = encoded.permute(0, 2, 1)
        encoded_feature = encoded.contiguous().view(B, hidden, h, w) #768,24,24
        fearure = transformer[2] #512,48,48; 256,96,96; 64,192,192
        return (fearure[0], fearure[1], fearure[2], encoded_feature)


class decoder(nn.Module):
    def __init__(self, x, num_class, feature_base=32, drop = 0.05):
        super(decoder, self).__init__()
        self.num_class = num_class
        if x =='50':
            self.deconv6 = bn_relu_deconv2d(c_i=64 * feature_base, c_o=32 * feature_base, k=2)
            self.res6 = ResidualBlock(in_channels=32 * feature_base, out_channels=32 * feature_base, kernel=3)
            self.conv6 = nn.Conv2d(in_channels=64 * feature_base, out_channels=32 * feature_base, kernel_size=3,
                                   padding='same')
            self.drop6 = nn.Dropout(drop)
            self.deconv7 = bn_relu_deconv2d(c_i=32 * feature_base, c_o=16 * feature_base, k=2)
            self.res7 = ResidualBlock(in_channels=16 * feature_base, out_channels=16 * feature_base, kernel=3)
            self.conv7 = nn.Conv2d(in_channels=32 * feature_base, out_channels=16 * feature_base, kernel_size=3,
                                   padding='same')
            self.drop7 = nn.Dropout(drop)
            self.deconv8 = bn_relu_deconv2d(c_i=16 * feature_base, c_o=8 * feature_base, k=2)
            self.res8 = ResidualBlock(in_channels=8 * feature_base, out_channels=8 * feature_base, kernel=3)
            self.conv8 = nn.Conv2d(in_channels=16 * feature_base, out_channels=8 * feature_base, kernel_size=3,
                                   padding='same')
            self.drop8 = nn.Dropout(drop)
            self.deconv9 = bn_relu_deconv2d(c_i=8 * feature_base, c_o=2 * feature_base, k=2)
            self.res9 = ResidualBlock(in_channels=feature_base, out_channels=feature_base, kernel=3)
            self.conv9 = nn.Conv2d(in_channels=4 * feature_base, out_channels=feature_base, kernel_size=3,
                                   padding='same')
            self.drop9 = nn.Dropout(drop)
            self.deconv10 = bn_relu_deconv2d(c_i=feature_base, c_o=feature_base, k=2)
            self.conv9_2 = bn_relu_conv2d(c_i=feature_base, c_o=self.num_class, k=3)
        else:
            self.deconv6 = bn_relu_deconv2d(c_i=16 * feature_base, c_o=8 * feature_base, k=2)
            self.res6 = ResidualBlock(in_channels=8 * feature_base, out_channels=8 * feature_base, kernel=3)
            self.conv6 = nn.Conv2d(in_channels=16 * feature_base, out_channels=8 * feature_base, kernel_size=3,
                                   padding='same')
            self.drop6 = nn.Dropout(drop)
            self.deconv7 = bn_relu_deconv2d(c_i=8 * feature_base, c_o=4 * feature_base, k=2)
            self.res7 = ResidualBlock(in_channels=4 * feature_base, out_channels=4 * feature_base, kernel=3)
            self.conv7 = nn.Conv2d(in_channels=8 * feature_base, out_channels=4 * feature_base, kernel_size=3,
                                   padding='same')
            self.drop7 = nn.Dropout(drop)
            self.deconv8 = bn_relu_deconv2d(c_i=4 * feature_base, c_o=2 * feature_base, k=2)
            self.res8 = ResidualBlock(in_channels=2 * feature_base, out_channels=2 * feature_base, kernel=3)
            self.conv8 = nn.Conv2d(in_channels=4 * feature_base, out_channels=2 * feature_base, kernel_size=3,
                                   padding='same')
            self.drop8 = nn.Dropout(drop)
            self.deconv9 = bn_relu_deconv2d(c_i=2 * feature_base, c_o= 2 * feature_base, k=2)
            self.res9 = ResidualBlock(in_channels=feature_base, out_channels=feature_base, kernel=3)
            self.conv9 = nn.Conv2d(in_channels=4 * feature_base, out_channels=feature_base, kernel_size=3,
                                   padding='same')
            self.drop9 = nn.Dropout(drop)
            self.deconv10 = bn_relu_deconv2d(c_i= feature_base, c_o= feature_base, k=2)
            self.conv9_2 = bn_relu_conv2d(c_i=feature_base, c_o=self.num_class, k=3)


    def forward(self, x):
        res1, res2, res3, res4, res5_2 = x
        deconv6 = self.deconv6(res5_2)
        sum6 = torch.cat((res4, deconv6), dim=1)
        conv6 = self.drop6(self.conv6(sum6))
        res6 = self.res6(conv6)

        deconv7 = self.deconv7(res6)
        sum7 = torch.cat((res3,deconv7), dim=1)
        conv7 = self.drop7(self.conv7(sum7))
        res7 = self.res7(conv7)

        deconv8 = self.deconv8(res7)
        sum8 = torch.cat((res2, deconv8), dim=1)
        conv8 = self.drop8(self.conv8(sum8))
        res8 = self.res8(conv8)

        deconv9 = self.deconv9(res8)
        sum9 = torch.cat((res1, deconv9), dim=1)
        conv9 = self.drop9(self.conv9(sum9))
        res9 = self.res9(conv9)

        deconv10 = self.deconv10(res9)
        output = self.conv9_2(deconv10)
        return output


class VIT_decoder(nn.Module):
    def __init__(self, num_class, feature_base=32, drop = 0.05):
        super(VIT_decoder, self).__init__()
        self.num_class = num_class
        self.deconv6 = bn_relu_deconv2d(c_i=24 * feature_base, c_o=16 * feature_base, k=2)
        self.res6 = ResidualBlock(in_channels=16 * feature_base, out_channels=16 * feature_base, kernel=3)
        self.conv6 = nn.Conv2d(in_channels=32 * feature_base, out_channels=16 * feature_base, kernel_size=3,
                               padding='same')
        self.drop6 = nn.Dropout(drop)
        self.deconv7 = bn_relu_deconv2d(c_i=16 * feature_base, c_o= 8 * feature_base, k=2)
        self.res7 = ResidualBlock(in_channels=8 * feature_base, out_channels=8 * feature_base, kernel=3)
        self.conv7 = nn.Conv2d(in_channels=16 * feature_base, out_channels=8 * feature_base, kernel_size=3,
                               padding='same')
        self.drop7 = nn.Dropout(drop)
        self.deconv8 = bn_relu_deconv2d(c_i=8 * feature_base, c_o=2 * feature_base, k=2)
        self.res8 = ResidualBlock(in_channels= feature_base, out_channels=feature_base, kernel=3)
        self.conv8 = nn.Conv2d(in_channels=4 * feature_base, out_channels= feature_base, kernel_size=3,
                               padding='same')
        self.drop8 = nn.Dropout(drop)
        self.deconv10 = bn_relu_deconv2d(c_i=feature_base, c_o=feature_base, k=2)
        self.conv9_2 = bn_relu_conv2d(c_i=feature_base, c_o=self.num_class, k=3)


    def forward(self, x):
        decoder_feats = []
        res3, res2, res1, res4 = x
        deconv6 = self.deconv6(res4)
        sum6 = torch.cat((res3, deconv6), dim=1)
        conv6 = self.drop6(self.conv6(sum6))
        res6 = self.res6(conv6)
        decoder_feats.append(res6)
        
        deconv7 = self.deconv7(res6)
        sum7 = torch.cat((res2,deconv7), dim=1)
        conv7 = self.drop7(self.conv7(sum7))
        res7 = self.res7(conv7)
        decoder_feats.append(res7)

        deconv8 = self.deconv8(res7)
        sum8 = torch.cat((res1, deconv8), dim=1)
        conv8 = self.drop8(self.conv8(sum8))
        res8 = self.res8(conv8)
        deconv10 = self.deconv10(res8)
        decoder_feats.append(deconv10)
        
        output = self.conv9_2(deconv10)
        decoder_feats.append(res4)
        return output, decoder_feats


class ms_net(nn.Module):
    def __init__(self):
        super(ms_net, self).__init__()
        self.encoder = encoder()
        self.teacher_decoder_1 = decoder(num_class=2)
        self.teacher_decoder_2 = decoder(num_class=2)
        self.teacher_decoder_3 = decoder(num_class=2)
        self.student_decoder = decoder(num_class=2)

    def forward(self, x_1, x_2, x_3, train_student = False):
        feature_1 = self.encoder(x_1)
        feature_2 = self.encoder(x_2)
        feature_3 = self.encoder(x_3)

        if train_student:
            student_1 = self.student_decoder(feature_1)
            student_2 = self.student_decoder(feature_2)
            student_3 = self.student_decoder(feature_3)
            return student_1, student_2, student_3
        else:
            teacher_1 = self.teacher_decoder_1(feature_1)
            teacher_2 = self.teacher_decoder_2(feature_2)
            teacher_3 = self.teacher_decoder_3(feature_3)
            return teacher_1, teacher_2, teacher_3

class ms_IL_net(nn.Module):

    def __init__(self):
        super(ms_IL_net, self).__init__()
        self.encoder = encoder()
        self.teacher_decoder = decoder(num_class=2)
        self.student_decoder = decoder(num_class=2)

    def forward(self, x):
        feature = self.encoder(x)
        teacher = self.teacher_decoder(feature)
        student = self.student_decoder(feature)
        return teacher, student
    
class MS_net(nn.Module):
    
    def __init__(self, x):
        super(MS_net, self).__init__()
        self.encoder = encoder()
        self.teacher_decoder = decoder(x, num_class=4)

    def forward(self, x):
        feature = self.encoder(x)
        teacher = self.teacher_decoder(feature)
        return teacher

class Res_IL_net(nn.Module):

    def __init__(self, x):
        super(Res_IL_net, self).__init__()
        self.encoder = res_encoder(x)
        # if keep the encoder
        # for p in self.parameters():
        #     p.requires_grad = False
        self.teacher_decoder = decoder(x, num_class=2)
        self.student_decoder = decoder(x, num_class=2)

    def forward(self, x):
        feature = self.encoder(x)
        teacher = self.teacher_decoder(feature)
        student = self.student_decoder(feature)
        return teacher, student
    

class Res_net(nn.Module):

    def __init__(self, x, num_class=4):
        super(Res_net, self).__init__()
        self.encoder = res_encoder(x)
        # if keep the encoder
        # for p in self.parameters():
        #     p.requires_grad = False
        self.teacher_decoder = decoder(x, num_class=num_class)

    def forward(self, x):
        feature = self.encoder(x)
        teacher = self.teacher_decoder(feature)
        return teacher, feature[-1]
    

class VIT_IL_net(nn.Module):

    def __init__(self):
        super(VIT_IL_net, self).__init__()
        self.encoder = ViT_encoder()
        # if keep the encoder
        # for p in self.parameters():
        #     p.requires_grad = False
        self.teacher_decoder = VIT_decoder(num_class=4)
        self.student_decoder = VIT_decoder(num_class=4)

    def forward(self, x):
        feature = self.encoder(x)
        teacher = self.teacher_decoder(feature)
        student = self.student_decoder(feature)
        return teacher, student
    
class VIT_net(nn.Module):
    
    def __init__(self, num_class=4):
        super(VIT_net, self).__init__()
        self.encoder = ViT_encoder()
        # if keep the encoder
        # for p in self.parameters():
        #     p.requires_grad = False
        self.decoder = VIT_decoder(num_class=num_class)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feature = self.encoder(x)
        teacher, feats = self.decoder(feature)
        return teacher, feats


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, x, y):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])

if __name__ == "__main__":

    x1 = torch.rand(5, 768, 24, 24).cuda()
    x2 = torch.rand(5, 64, 192, 192).cuda()
    x3 = torch.rand(5, 256, 96, 96).cuda()
    x4 = torch.rand(5, 512, 48, 48).cuda()
    x = torch.rand(4, 3, 256, 256).cuda()
    # ms net test
    # net = ms_net().cuda()
    # y1, y2, y3 = net(x1, x2, x3)
    # print(y1.shape, y2.shape, y3.shape)
    # z1, z2, z3 = net(x1, x2, x3, True)
    # print(z1.shape, z2.shape, z3.shape)

    # res18_encoder test
    # net = res18_encoder().cuda()
    # y1 = net(x1)
    # print(y1.shape)
    #
    # Res_IL_net test
    # net = MS_net('18').cuda()
    # y1 = net(x)
    # print(y1.shape)
    net = UNet(in_chns=3, class_num=4).cuda()
    y1, latent = net(x)
    print(y1.shape, latent.shape)
    
    
    net = Res_net('50').cuda()
    y1, latent = net(x)
    print(y1.shape, latent.shape)
    
    net = Res_net('18').cuda()
    y1, latent = net(x)
    print(y1.shape, latent.shape)
    
    net = Res_net('34').cuda()
    y1, latent = net(x)
    print(y1.shape, latent.shape)

    net = VIT_net().cuda()
    y1, latent = net(x)
    print(y1.shape, latent.shape) 
