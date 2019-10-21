import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
from layers import *


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

class BasicRFB_b(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicRFB_b, self).__init__()
        self.out_channels = out_planes
        inter_planes = in_planes // 2
        self.single_branch = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1))
                )

    def forward(self, x):
        out = self.single_branch(x)
        return out



class RFBNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.base = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)

        # # full attention
        # self.attention_t1 = G_attention(256, 256, 3)
        # self.attention_t2 = G_attention(256, 256, 5)
        # self.attention_t3 = G_attention(256, 512, 10)
        # self.attention_t4 = G_attention(512, 1024, 19)
        # self.attention_t5 = G_attention(1024, 512, 38)
        #
        # self.BasicResConv5 = BasicResConv(256)
        # self.BasicResConv4 = BasicResConv(256)
        # self.BasicResConv3 = BasicResConv(512)
        # self.BasicResConv2 = BasicResConv(1024)
        # self.BasicResConv1 = BasicResConv(512)

################################################################################
        # generate new_sources_with fusion
        self.conv75_38 = BasicConv(256, 512, kernel_size=1)
        self.conv38 = BasicConv(512,512,kernel_size=1)
        self.conv19_38 = BasicConv(1024,512,kernel_size=1)

        self.conv38_19 = BasicConv(512,1024,kernel_size=1)
        self.conv19 = BasicConv(1024,1024,kernel_size=1)
        self.conv10_19 = BasicConv(512,1024,kernel_size=1)

        self.conv19_10 = BasicConv(1024,512,kernel_size=1)
        self.conv10 = BasicConv(512,512,kernel_size=1)
        self.conv5_10 = BasicConv(256,512,kernel_size=1)

        self.conv10_5 = BasicConv(512,256,kernel_size=1)
        self.conv5 = BasicConv(256,256,kernel_size=1)
        self.conv3_5 = BasicConv(256,256,kernel_size=1)


        # half attention
        self.attention_t5_19 = G_attention(256, 1024, 19)
        self.attention_t10_38 = G_attention(512, 512, 38)
        #
        # # half add conv
        self.BasicResConv19 = BasicResConv(1024)
        self.BasicResConv38 = BasicResConv(512)
        self.BasicResConv10 = BasicResConv(512)
        self.BasicResConv5 = BasicResConv(256)

        self.BasicConv10 = BasicConv(512,512,kernel_size=1)
        self.BasicConv5 = BasicConv(1024,256,kernel_size=1)
        # self.Pool10 = torch.nn.AvgPool2d(kernel_size=4,stride=4,padding=1)
        # self.Pool5 = torch.nn.AvgPool2d(kernel_size=4,stride=4,padding=1)

        self.BasicConv38_loc = Basic2conv(512,512)
        self.BasicConv19_loc = Basic2conv(1024,512)
        self.BasicConv10_loc = Basic2conv(512,512)
        self.BasicConv5_loc = Basic2conv(256,512)
        self.BasicConv3_loc = Basic2conv(256,512)
        self.BasicConv1_loc = Basic2conv(256,512)

        self.BasicConv38_conf = Basic2conv(512,512)
        self.BasicConv19_conf = Basic2conv(1024,512)
        self.BasicConv10_conf = Basic2conv(512,512)
        self.BasicConv5_conf = Basic2conv(256,512)
        self.BasicConv3_conf = Basic2conv(256,512)
        self.BasicConv1_conf = Basic2conv(256,512)
###############################################################################33

        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax()

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:attention_t5
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources_init = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)
            if k==16:
                source_38 = x
        s = self.L2Norm(x)
        sources_init.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        sources_init.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator - 1 or k % 2 == 1:
                sources_init.append(x)
#########################################################################
        # generate attention
        sources = []
        #generate 38
        feat_75_38 = self.conv75_38(source_38)
        feat_19_38 = F.upsample_bilinear(self.conv19_38(sources_init[1]),size=38)
        # feat_38 = self.conv38(sources_init[0])
        s_38 = feat_19_38 + sources_init[0]+feat_75_38
        sources.append(s_38)

        feat_38_19 = F.adaptive_avg_pool2d(self.conv38_19(sources_init[0]),19)
        feat_10_19 = F.upsample_bilinear(self.conv10_19(sources_init[2]),size=19)
        # feat_19 = self.conv19(sources_init[1])
        s_19 = feat_38_19 + feat_10_19 + sources_init[1]
        sources.append(s_19)

        feat_19_10 = F.adaptive_avg_pool2d(self.conv19_10(sources_init[1]),10)
        # feat_10 = self.conv10(sources_init[2])
        # feat_5_10 = F.upsample_bilinear(self.conv5_10(sources_init[3]),size=10)
        s_10 = feat_19_10 + sources_init[2]
        sources.append(s_10)

        feat_10_5 = F.adaptive_avg_pool2d(self.conv10_5(sources_init[2]),5)
        # feat_5 = self.conv5(sources_init[3])
        feat_3_5 = F.upsample_bilinear(self.conv3_5(sources_init[4]),size=5)
        s_5 = sources_init[3] + feat_10_5 + feat_3_5
        sources.append(s_5)

        s_3 = sources_init[4]
        sources.append(s_3)

        s_1 = sources_init[5]
        sources.append(s_1)

##########################################################################33
        # generate new_sources_with fusion


        # # half attention skip with conv
        att3 = self.attention_t5_19(sources[-3])
        att4 = self.attention_t10_38(sources[-4])

        y0 = torch.mul(sources[0],1-att4)
        y0 = self.BasicResConv38(y0)
        y0_loc = self.BasicConv38_loc(y0)
        y0_conf = self.BasicConv38_conf(y0)

        y1 = torch.mul(sources[1], 1-att3)
        y1 = self.BasicResConv19(y1)
        y1_loc = self.BasicConv19_loc(y1)
        y1_conf = self.BasicConv19_conf(y1)

        y0_res = torch.mul(sources[0],att4)
        y0_res = self.BasicConv10(y0_res)
        y1_res = torch.mul(sources[1], att3)
        y1_res = self.BasicConv5(y1_res)

        y2 = sources[2] + F.adaptive_avg_pool2d(y0_res,10)
        y2 = self.BasicResConv10(y2)
        y2_loc = self.BasicConv10_loc(y2)
        y2_conf = self.BasicConv10_conf(y2)

        y3 = sources[3] + F.adaptive_avg_pool2d(y1_res,5)
        y3 = self.BasicResConv5(y3)
        y3_loc = self.BasicConv5_loc(y3)
        y3_conf = self.BasicConv5_conf(y3)

        y4 = sources[4]
        y4_loc = self.BasicConv3_loc(y4)
        y4_conf = self.BasicConv3_conf(y4)

        y5 = x
        y5_loc = self.BasicConv1_loc(y5)
        y5_conf = self.BasicConv1_conf(y5)

        new_sources_conf = []
        new_sources_conf.append(y0_conf)
        new_sources_conf.append(y1_conf)
        new_sources_conf.append(y2_conf)
        new_sources_conf.append(y3_conf)
        new_sources_conf.append(y4_conf)
        new_sources_conf.append(y5_conf)

        new_sources_loc = []
        new_sources_loc.append(y0_loc)
        new_sources_loc.append(y1_loc)
        new_sources_loc.append(y2_loc)
        new_sources_loc.append(y3_loc)
        new_sources_loc.append(y4_loc)
        new_sources_loc.append(y5_loc)


###################################################################33
        # apply multibox head to source layers
        for (x_loc, x_conf, l, c) in zip(new_sources_loc, new_sources_conf, self.loc, self.conf):
            loc.append(l(x_loc).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x_conf).permute(0, 2, 3, 1).contiguous())

        #print([o.size() for o in loc])

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
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
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def add_extras(size, cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [BasicRFB_b(in_channels, cfg[k+1], stride=2)]
                else:
                    layers += [BasicRFB_b(in_channels, cfg[k+1], stride=2)]
        in_channels = v
    if size == 512:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=4,stride=1,padding=1)]
    elif size ==300:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers

extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
}

def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [1, -2]
    groups = 9
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers +=[nn.Conv2d(512,
                                 cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(512,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    i = 2
    indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator-1 or k % 2 == 1:
            loc_layers += [nn.Conv2d(512, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(512, cfg[i]
                                  * num_classes, kernel_size=3, padding=1)]
            i +=1
    return vgg, extra_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return RFBNet(phase, size, *multibox(size, vgg(base[str(size)], 3),
                                add_extras(size, extras[str(size)], 1024),
                                mbox[str(size)], num_classes), num_classes)
