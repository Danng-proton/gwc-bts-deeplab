from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
from math import log
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from scipy import sparse as sp


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x=x.repeat(1,3,1,1)
        x = self.firstconv(x)
        #x[0,13,:,:]=0
        #print(x[:,14,:,:])
        l1 = self.layer1(x)
        
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)

        #l3[0,0,:,:]=0
        l4 = self.layer4(l3)      
        
        gwc_feature = torch.cat((l2, l3, l4), dim=1)
        fea_list=[x,l1,l2,l3,l4]
        # print(x.shape)
        # print(l1.shape)
        # print(l4.shape) 

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
           # print("!!!!!!!!!!")
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature,"feature":fea_list}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class GwcNet(nn.Module):
    def __init__(self,args, use_concat_volume=True):
        super(GwcNet, self).__init__()
        self.maxdisp = args.maxdisp
        self.r2l=args.r2l
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        for m in self.modules():
            # m.weight.requires_grad=False
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.weight.requires_grad=False
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.weight.requires_grad=False
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.requires_grad=True
                #m.bias.requires_grad=True
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                #m.weight.requires_grad=True
                #m.bias.requires_grad=True
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
               # m.bias.requires_grad=False

    def forward(self, left, right):
        #left_edge=y_gradient_1order(x_gradient_1order(left))
        #right_edge=y_gradient_1order(x_gradient_1order(right))
        #mask=left_edge>0.5
        #print(left,right)
        #right=right/2.0
        #pred_tra=mutual_info_pred(left,right,self.maxdisp)
        #print("left")
        features_left = self.feature_extraction(left)
        #print("right")
        features_right = self.feature_extraction(right)

        if self.r2l==False:
            gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        else:
            gwc_volume = build_r2l_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        # add by yyx
        left_fea=features_left["feature"]
        right_fea=features_right["feature"]
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.training:
        #if True:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            return [pred0, pred1, pred2, pred3]

        else:
            cost3 = self.classif3(out3)
            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            # _,d,H,W=pred3.size()
            # for h in range(0,H,10):
            #     for w in range(0,W,20):
            #         point_disp=pred3[0,:,h,w]
            #         distribution=point_disp.view(-1)
            #         plt.xlabel('Disparity')
            #         plt.ylabel('Probability')
            #         x=np.arange(len(distribution))
            #         print(h+1,"_",w+1,distribution)
            #         plt.xlim(xmax=192, xmin=0)
            #         plt.ylim(ymax=1,ymin=0)
                    
            #         #plt.hist(x=distribution, bins=d, color='#0504aa',alpha=0.7, rwidth=0.9)
            #         plt.bar(x, distribution,fc='r')
            #         filename='fea_map/kitti/disparity_distribution_4/'+str(h+1)+'_'+str(w+1)+'.png'
            #         plt.savefig(filename)
            #         plt.close()
            confidence_var=torch.var(pred3,dim=1)
            confidence,index=torch.max(pred3,dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred3],confidence,confidence_var,index
            #return pred_tra


def GwcNet_G(d):
    
    return GwcNet(d, use_concat_volume=False)


def GwcNet_GC(d):

    return GwcNet(d, use_concat_volume=True)



# def entropy(labels):
#
#     if len(labels) == 0:
#         return 1.0
#     label_idx = np.unique(labels, return_inverse=True)[1]
#     pi = np.bincount(label_idx).astype(np.float64)
#     #print("!!!",pi)
#     #pi = pi[pi > 0]
#     pi_sum = np.sum(pi)
#     # log(a / b) should be calculated as log(a) - log(b) for
#     # possible loss of precision
#     entro=-((pi / pi_sum) * (np.log(pi) - log(pi_sum)))
#     entro = np.where(np.isnan(entro), 0, entro)
#     return entro
def entropy(pi):
    #pi = pi[pi >= 0]
    #print(pi)
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    join_entropy=-((pi / pi_sum) * (np.log(pi) - log(pi_sum)))
    #print(join_entropy)
    #zero=np.zeros([1,6,6])
    join_entropy=np.where(np.isnan(join_entropy),0,join_entropy)
    #join_entropy[0, 0] = 0
    return join_entropy
def mutual_entropy(a,b,bin_num):
    a=np.array(a)
    b=np.array(b)
    #print(a.shape)
    a_hist=np.histogram(a,bins=bin_num+1,range=(0,bin_num))[0]
    b_hist = np.histogram(b, bins=bin_num+1, range=(0, bin_num))[0]
    #print("!!!!!",a.shape,a_hist.shape)
    #a = a.reshape(1,-1)
    #b = b.reshape(1,-1)
    # print(a.ravel().shape,b.ravel().shape)
    ab_hist=np.histogram2d(a.ravel(),b.ravel(),bins=(bin_num+1,bin_num+1),range=[(0,bin_num),(0,bin_num)])[0]
    # print(ab_hist.shape)
    # print("a entropy",entropy(a_hist))
    # print("b entropy",entropy(b_hist))
    # print("ab entropy",entropy(ab_hist)[7])
    # a_entropy=np.repeat(entropy(a_hist),[bin_num],axis=0)
    # b_entropy = np.repeat(entropy(b_hist),bin_num,axis=0)
    # print(a_entropy.shape)
    # mutual_entro=np.zeros([bin_num,bin_num])
    # for b in range(bin_num):
    #     mutual_entro[b]=entropy(a_hist)+entropy(b_hist)-(entropy(ab_hist))[b]
    mutual_entro=[entropy(a_hist),entropy(b_hist),entropy(ab_hist)]
    #print("!!",mutual_entro[0].shape)
    return mutual_entro

def mutual_info_pred(left, right,maxdisp):
    # left=(left+0.5).floor()
    # right=(right+0.5).floor()
    #print(left,right)
    mutual_info=mutual_entropy(left,right,255)
    B, C, H, W = left.shape
    volume = left.new_zeros([B, maxdisp, H, W])
    for x in range(H):
        for y in range(W):
            for i in range(maxdisp):
                if x<=i:
                    volume[:,i,x,y]=0
                else:
                    left_i=int(left[0,0,x,y])
                    right_i=int(right[0,0,x-i,y])
                    # print(left_i,right_i)
                    #print(mutual_info[0].shape)
                    volume[0,i,x,y]=mutual_info[0][left_i]+mutual_info[1][right_i]-mutual_info[2][left_i,right_i]
                #print(volume[0,i,x,y])
    volume=torch.softmax(volume,dim=1)
    confidence,pred=torch.max(volume,dim=1)
    pred=pred.float()
    pred_sub=disparity_regression(volume,maxdisp)
    return pred

def x_gradient_1order(img):
    img = img.permute(0,2,3,1)
    img_l = img[:,:,1:,:] - img[:,:,:-1,:]
    img_r = img[:,:,-1,:] - img[:,:,-2,:]
    img_r = img_r.unsqueeze(2)
    img  = torch.cat([img_l, img_r], 2).permute(0, 3, 1, 2)
    return img

def y_gradient_1order(img):
    # pdb.set_trace()
    img = img.permute(0,2,3,1)
    img_u = img[:,1:,:,:] - img[:,:-1,:,:]
    img_d = img[:,-1,:,:] - img[:,-2,:,:]
    img_d = img_d.unsqueeze(1)
    img  = torch.cat([img_u, img_d], 1).permute(0, 3, 1, 2)
    return img

def gradient_1order(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    return xgrad
