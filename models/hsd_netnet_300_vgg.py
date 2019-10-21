# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from models.model_helper import weights_init
from models.attention import PAM_Module



class BasicResConv(nn.Module):

    def __init__(self, in_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicResConv, self).__init__()
        # self.out_channels = out_planes
        inter_planes = in_planes//4
        self.res_branch = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1,padding=0, groups=groups,relu=relu, bn=bn, bias=bias),
                BasicConv(inter_planes, inter_planes, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation,groups=groups,relu=relu, bn=bn, bias = bias),
                BasicConv(inter_planes, in_planes, kernel_size=1, stride=1, padding=0, groups=groups, relu=False, bn=bn, bias=bias)
                )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.res_branch(x)
        output = x + x1
        output = self.relu(output)
        return output

class G_attention(nn.Module):

    def __init__(self, in_planes, out_planes, re_size, kernel_size=1, stride=1,padding=0, dilation=1, groups=1,bias=False):
        super(G_attention,self).__init__()
        self.out_channels = out_planes
        self.int_channels = in_planes
        self.re_size = re_size
        # self.up_sample = nn.functional.upsample_bilinear(size=[re_size,re_size])
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.up_sample(x)
        x = nn.functional.upsample_bilinear(x, size=self.re_size)
        x = self.conv(x)
        x = self.sigmoid(x)
        # x = 1 - x
        return x

class Basic2conv(nn.Module):

    def __init__(self, in_planes, out_planes=256, stride=1):
        super(Basic2conv, self).__init__()
        self.out_channels = out_planes
        inter_planes = out_planes
        self.single_branch = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, out_planes, kernel_size=1, stride=1, padding=0)
                )

    def forward(self, x):
        out = self.single_branch(x)
        return out


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(
            x) * x
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.single_branch = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=2, dilation=2),
            BasicConv(inter_planes, out_planes, kernel_size=(3, 3), stride=1, padding=(1, 1))
        )

    def forward(self, x):
        out = self.single_branch(x)
        return out
# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [
        pool5, conv6,
        nn.ReLU(inplace=True), conv7,
        nn.ReLU(inplace=True)
    ]
    return layers


base = {
    '300': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512
    ],
    '512': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
        512, 512, 512
    ],
}


def add_extras(size):
    layers = []
    layers += [BasicBlock(1024, 256, stride=2)]
    layers += [BasicBlock(256, 256, stride=2)]
    return layers

#

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * (1+y)

class FEModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=nn.BatchNorm2d):
        super(FEModule, self).__init__()
        self.out_channels = out_channels
        inter_channels = in_channels // 4
        self.brancha = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.brancha1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                     norm_layer(inter_channels),
                                     nn.ReLU())

        # aspp
        self.sl = nn.Sequential(
            BasicConv(in_channels, inter_channels, kernel_size=1, stride=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        )
        self.sn = nn.Sequential(
            BasicConv(in_channels, inter_channels, kernel_size=1, stride=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        )

        self.fuse = nn.Sequential(nn.Dropout2d(0.1, False),
                                     nn.Conv2d(inter_channels + inter_channels + inter_channels, out_channels,
                                               kernel_size=3, stride=stride, padding=1, bias=False),
                                     norm_layer(out_channels),
                                     nn.ReLU())



    def forward(self, x):
        sa_feat = self.sa(self.brancha(x))
        sa_conv = self.brancha1(sa_feat)

        sl_output = self.sl(x)
        sn_output = self.sn(x)

        feat_cat = torch.cat([sa_conv, sl_output, sn_output], dim=1)
        sasc_output = self.fuse(feat_cat)

        return sasc_output

def trans_head():
    arm_trans1 = []
    arm_trans1 += [BasicConv(512, 256, kernel_size=1, stride=1, padding=0)]
    arm_trans1 += [BasicConv(1024, 256, kernel_size=1, stride=1, padding=0)]
    arm_trans1 += [BasicConv(256, 256, kernel_size=1, stride=1, padding=0)]
    arm_trans1 += [BasicConv(256, 256, kernel_size=1, stride=1, padding=0)]

    # arm_trans = []
    # arm_trans += [BasicConv(512, 256, kernel_size=3, stride=1, padding=1)]
    # arm_trans += [BasicConv(1024, 256, kernel_size=3, stride=1, padding=1)]
    # arm_trans += [BasicConv(256, 256, kernel_size=3, stride=1, padding=1)]
    # arm_trans += [BasicConv(256, 256, kernel_size=3, stride=1, padding=1)]

    orm_trans = []
    orm_trans += [BasicConv(256, 512, kernel_size=3, stride=1, padding=1)]
    orm_trans += [BasicConv(256, 512, kernel_size=3, stride=1, padding=1)]
    orm_trans += [BasicConv(256, 512, kernel_size=3, stride=1, padding=1)]
    orm_trans += [BasicConv(256, 256, kernel_size=3, stride=1, padding=1)]

    return arm_trans1, orm_trans

class VGG16Extractor(nn.Module):
    def __init__(self, size, channel_size='48'):
        super(VGG16Extractor, self).__init__()
        self.vgg = nn.ModuleList(vgg(base[str(size)], 3))
        self.extras = nn.ModuleList(add_extras(str(size)))

        self.fe1 = FEModule(256,256)
        self.fe2 = FEModule(256,256)
        self.fe3 = FEModule(256,256)
        self.arm_trans = nn.ModuleList(trans_head()[0])
        self.orm_trans = nn.ModuleList(trans_head()[1])

        ################# using our netnet
        ################################################################################
        # generate new_sources_with fusion
        self.conv75_38 = BasicConv(256, 512, kernel_size=1)
        self.conv38 = BasicConv(512, 512, kernel_size=1)
        self.conv19_38 = BasicConv(1024, 512, kernel_size=1)

        self.conv38_19 = BasicConv(512, 1024, kernel_size=1)
        self.conv19 = BasicConv(1024, 1024, kernel_size=1)
        self.conv10_19 = BasicConv(256, 1024, kernel_size=1)

        self.conv19_10 = BasicConv(1024, 256, kernel_size=1)
        self.conv10 = BasicConv(256, 256, kernel_size=1)
        self.conv5_10 = BasicConv(256, 256, kernel_size=1)

        self.conv10_5 = BasicConv(256, 256, kernel_size=1)
        self.conv5 = BasicConv(256, 256, kernel_size=1)

        self.size0 = 40
        self.size1 = 20
        self.size2 = 10
        self.size3 = 5

        # half attention
        self.attention_t5_19 = G_attention(256, 1024, self.size1)
        self.attention_t10_38 = G_attention(256, 512, self.size0)
        #
        # # half add conv
        self.BasicResConv19 = BasicResConv(1024)
        self.BasicResConv38 = BasicResConv(512)
        self.BasicResConv10 = BasicResConv(256)
        self.BasicResConv5 = BasicResConv(256)

        self.BasicConv10 = BasicConv(512, 256, kernel_size=1)
        self.BasicConv5 = BasicConv(1024, 256, kernel_size=1)
        ###############################################



        ####


        self._init_modules()

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.orm_trans.apply(weights_init)
        self.arm_trans.apply(weights_init)
        self.fe1.apply(weights_init)
        self.fe2.apply(weights_init)
        self.fe3.apply(weights_init)


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        arm_sources_init = list()
        arm_sources = list()

        for i in range(23):
            x = self.vgg[i](x)
            if i == 16:
                source_38 = x
        #38x38
        c2 = x
        print(c2.shape)
        # c2 = self.arm_trans[0](c2)
        arm_sources_init.append(c2)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        #19x19
        c3 = x
        print(c3.shape)
        # c3 = self.arm_trans[1](c3)
        arm_sources_init.append(c3)

        # 10x10
        x = self.extras[0](x)
        print(x.shape)
        # c4 = x
        # c4 = self.arm_trans[2](x)
        arm_sources_init.append(x)

        # 5x5
        x = self.extras[1](x)
        print(x.shape)
        # c5 = x
        # c5 = self.arm_trans[3](x)
        arm_sources_init.append(x)

        #########################################################################
        # generate attention
        sources = []
        # generate 38
        feat_75_38 = self.conv75_38(source_38)
        feat_19_38 = F.upsample_bilinear(self.conv19_38(arm_sources_init[1]), size=self.size0)
        # feat_38 = self.conv38(sources_init[0])
        s_38 = feat_19_38 + arm_sources_init[0] + feat_75_38
        sources.append(s_38)

        feat_38_19 = F.adaptive_avg_pool2d(self.conv38_19(arm_sources_init[0]), self.size1)
        feat_10_19 = F.upsample_bilinear(self.conv10_19(arm_sources_init[2]), size=self.size1)
        # feat_19 = self.conv19(sources_init[1])
        s_19 = feat_38_19 + feat_10_19 + arm_sources_init[1]
        sources.append(s_19)

        feat_19_10 = F.adaptive_avg_pool2d(self.conv19_10(arm_sources_init[1]), self.size2)
        # feat_10 = self.conv10(sources_init[2])
        feat_5_10 = F.upsample_bilinear(self.conv5_10(arm_sources_init[3]),size=self.size2)
        s_10 = feat_19_10 + arm_sources_init[2]+feat_5_10
        sources.append(s_10)

        feat_10_5 = F.adaptive_avg_pool2d(self.conv10_5(arm_sources_init[2]), self.size3)
        # feat_5 = self.conv5(sources_init[3])
        s_5 = arm_sources_init[3] + feat_10_5
        sources.append(s_5)

        ##########################################################################33
        # generate new_sources_with fusion


        # # half attention skip with conv
        att3 = self.attention_t5_19(sources[-1])
        att4 = self.attention_t10_38(sources[-2])

        y0 = torch.mul(sources[0], 1 - att4)
        y0 = self.BasicResConv38(y0)

        y1 = torch.mul(sources[1], 1 - att3)
        y1 = self.BasicResConv19(y1)

        y0_res = torch.mul(sources[0], att4)
        y0_res = self.BasicConv10(y0_res)
        y1_res = torch.mul(sources[1], att3)
        y1_res = self.BasicConv5(y1_res)

        y2 = sources[2] + F.adaptive_avg_pool2d(y0_res, self.size2)
        y2 = self.BasicResConv10(y2)

        y3 = sources[3] + F.adaptive_avg_pool2d(y1_res, self.size3)
        y3 = self.BasicResConv5(y3)

        arm_sources.append(self.arm_trans[0](y0))
        arm_sources.append(self.arm_trans[1](y1))
        arm_sources.append(self.arm_trans[2](y2))
        arm_sources.append(self.arm_trans[3](y3))
        ###########################################################################

        odm_sources = []
        # up = F.upsample(arm_sources[1], size=arm_sources[0].size()[2:], mode='bilinear')
        odm_sources.append(self.fe1(arm_sources[0]))
        # up = F.upsample(arm_sources[2], size=arm_sources[1].size()[2:], mode='bilinear')
        odm_sources.append(self.fe2(arm_sources[1]))
        # up = F.upsample(arm_sources[3], size=arm_sources[2].size()[2:], mode='bilinear')
        odm_sources.append(self.fe3(arm_sources[2]))
        odm_sources.append(self.orm_trans[3](arm_sources[3]))


        return arm_sources, odm_sources


def hsd_vgg(size):
    return VGG16Extractor(size)