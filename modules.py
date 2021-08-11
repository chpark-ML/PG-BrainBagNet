import torch
import torch.nn as nn
import setting_2 as fst
import numpy as np
import utils as ut
import setting as st
import math
from torch.autograd import Function
import torch.nn.functional as F
import nibabel as nib
import torch
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable
import random
from collections import namedtuple
from torch.distributions import Categorical

def MC_dropout(act_vec, p=0.2, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=False)

class LayerNorm(nn.Module):
    def __init__(self, dim, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma.unsqueeze(-1) * x + self.beta.unsqueeze(-1)

class LayerNorm3d(nn.Module):
    def __init__(self, dim, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        # u = nn.AvgPool3d(kernel_size=5, stride=1, padding=2)(x)
        u = x.mean(1, keepdim=True)
        # s = nn.AvgPool3d(kernel_size=5, stride=1, padding=2)((x - u).pow(2)).mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x + self.beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class InputDependentCombiningWeights_v2(nn.Module):
    def __init__(self, in_plance, spatial_rank= 1):
        super(InputDependentCombiningWeights_v2, self).__init__()
        """ 1 """
        self.dim_reduction_layer = nn.Conv3d(in_plance, spatial_rank, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)

        """ 2 """
        # self.dilations = [1, 1, 1, 1]
        self.dilations = [1, 2, 4, 8]
        self.multiscale_layers = nn.ModuleList([])
        for i in range(len(self.dilations)):
            self.multiscale_layers.append(nn.Conv3d(spatial_rank, spatial_rank, kernel_size=3, stride=1, padding=0, dilation=self.dilations[i], groups=spatial_rank, bias=True))

        """ 3 """
        ch = spatial_rank * (len(self.dilations) + 2)
        self.squeeze_layer = nn.Sequential(
            nn.Conv3d(ch, ch//2, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.ReLU(inplace=True),
        )

        """ 4 """
        self.excite_layer = nn.Sequential(
            nn.Conv3d(ch//2, ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.Sigmoid(),
        )

        """ 5 """
        self.proj_layer = nn.Conv3d(ch, spatial_rank, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)

    def forward(self, input_tensor, size):
        x_lowd = self.dim_reduction_layer(input_tensor)  # batch, 16, 85, 105, 80
        x_pool = nn.AvgPool3d(kernel_size=x_lowd.size()[-3:], stride=1)(x_lowd)

        x_multiscale = [
            F.interpolate(x_lowd, size=size, mode='trilinear', align_corners=False),
            F.interpolate(x_pool, size=size, mode='trilinear', align_corners=False),
        ]

        for r, layer in zip(self.dilations, self.multiscale_layers):
            x_multiscale.append(
                F.interpolate(layer(x_lowd), size=size, mode='trilinear', align_corners=False),
            )

        x_multiscale = torch.cat(x_multiscale, 1)
        x_0 = self.squeeze_layer(x_multiscale)
        x_0 = self.excite_layer(x_0)
        x_0 = self.proj_layer(x_0)
        return x_0

class Input_Dependent_LRLC_v2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1, act_func='relu', norm_layer='in', bias=False, n_K = 1):
        super(Input_Dependent_LRLC_v2, self).__init__()
        self.n_K = n_K

        self.cnn_layers = nn.Conv3d(in_channels, out_channels * n_K, kernel_size, stride, padding, dilation, padding_mode='replicate', groups=groups, bias=bias)

        if norm_layer == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm_layer is None:
            self.norm = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

        self.combining_weights_layer = InputDependentCombiningWeights_v2(in_channels, spatial_rank=n_K)


    def forward(self, input_tensor):
        """ weight """
        out = self.cnn_layers(input_tensor) # batch, out * rank, w, h, d (6, 64, 42, 52, 39)
        batch_dim, _, x_dim, y_dim, z_dim = out.shape
        weight = self.combining_weights_layer(input_tensor, size=(x_dim, y_dim, z_dim)) # batch, n_K, 42, 52, 39
        out = out.view(batch_dim, -1, self.n_K, x_dim, y_dim, z_dim) # batch, f, n_K, w, h, d
        weight = weight.unsqueeze(1)  # batch, 1, n_K, w, h, d
        f_out = torch.sum((out * weight), dim = 2) # batch, f, w, h, d

        if self.norm is not None:
            f_out = self.norm(f_out)

        f_out = self.act_func(f_out)
        return f_out

class InputDependentCombiningWeights(nn.Module):
    def __init__(self, in_plance, spatial_rank= 1):
        super(InputDependentCombiningWeights, self).__init__()
        """ 1 """
        self.dim_reduction_layer = nn.Conv3d(in_plance, spatial_rank, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        """ 2 """
        self.dilations = [1, 2, 4, 8]
        # self.dilations = [1, 2, 4]
        self.multiscale_layers = nn.ModuleList([])
        for i in range(len(self.dilations)):
            self.multiscale_layers.append(nn.Conv3d(spatial_rank, spatial_rank, kernel_size=3, stride=1, padding=0, dilation=self.dilations[i], groups=spatial_rank, bias=True))

        """ 3 """
        ch = spatial_rank * (len(self.dilations) + 2)
        self.squeeze_layer = nn.Sequential(
            nn.Conv3d(ch, ch//2, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.ReLU(inplace=True),
        )

        """ 4 """
        self.excite_layer = nn.Sequential(
            nn.Conv3d(ch//2, ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.Sigmoid(),
        )

        """ 5 """
        self.proj_layer = nn.Conv3d(ch, spatial_rank, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)

    def forward(self, input_tensor, size):
        x_lowd = self.dim_reduction_layer(input_tensor)  # batch, 16, 85, 105, 80
        x_pool = nn.AvgPool3d(kernel_size=x_lowd.size()[-3:], stride=1)(x_lowd)

        x_multiscale = [
            F.interpolate(x_lowd, size=size, mode='trilinear', align_corners=True),
            F.interpolate(x_pool, size=size, mode='trilinear', align_corners=True),
        ]

        for r, layer in zip(self.dilations, self.multiscale_layers):
            x_multiscale.append(
                F.interpolate(layer(x_lowd), size=size, mode='trilinear', align_corners=True),
            )

        x_multiscale = torch.cat(x_multiscale, 1)
        x_0 = self.squeeze_layer(x_multiscale)
        x_0 = self.excite_layer(x_0)
        x_0 = self.proj_layer(x_0)
        x_0 = nn.Sigmoid()(x_0)
        return x_0

class Input_Dependent_LRLC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1, act_func='relu', norm_layer='in', bias=False, n_K = 1):
        super(Input_Dependent_LRLC, self).__init__()
        self.n_K = n_K

        self.cnn_layers = nn.Conv3d(in_channels, out_channels * n_K, kernel_size, stride, padding, dilation, padding_mode='replicate', groups=groups, bias=bias)

        if norm_layer == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm_layer is None:
            self.norm = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

        self.combining_weights_layer = InputDependentCombiningWeights(in_channels, spatial_rank=n_K)

        #TODO : define the parameters
        # self.out_channels = out_channels # out_plane
        # self.shape_feature_map = np_feature_map_size # [w, h, d]
        # self.dim_feature_map = np_feature_map_size.shape[0] # 3

        """ zero initialized bias vectors for width, height, depth"""
        # self.list_parameter_b = nn.ParameterList([])
        # for i in range(self.dim_feature_map):
        #     # alpha = nn.Parameter(torch.zeros(np_feature_map_size[i]))
        #
        #     alpha = nn.Parameter(torch.Tensor(np_feature_map_size[i]))
        #     nn.init.kaiming_normal_(alpha, nonlinearity='relu')
        #
        #     self.list_parameter_b.append(alpha)
        #
        # # alpha = nn.Parameter(torch.zeros(out_channels))
        # alpha = nn.Parameter(torch.Tensor(out_channels))
        # nn.init.kaiming_normal_(alpha, nonlinearity='relu')
        # self.list_parameter_b.append(alpha)

    def forward(self, input_tensor):
        """ weight """
        out = self.cnn_layers(input_tensor) # batch, out * rank, w, h, d (6, 64, 42, 52, 39)
        batch_dim, _, x_dim, y_dim, z_dim = out.shape
        weight = self.combining_weights_layer(input_tensor, size=(x_dim, y_dim, z_dim)) # batch, n_K, 42, 52, 39
        out = out.view(batch_dim, -1, self.n_K, x_dim, y_dim, z_dim) # batch, f, n_K, w, h, d
        weight = weight.unsqueeze(1)  # batch, 1, n_K, w, h, d
        f_out = torch.sum((out * weight), dim = 2) # batch, f, w, h, d

        """ bias """
        # xx_range = self.list_parameter_b[0]
        # xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, x_dim)
        # xx_range = xx_range[:, None, :, None, None]
        #
        # yy_range = self.list_parameter_b[1]
        # yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, y_dim)  # (batch, 200)
        # yy_range = yy_range[:, None, None, :, None]
        #
        # zz_range = self.list_parameter_b[2]
        # zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, z_dim)  # (batch, 200)
        # zz_range = zz_range[:, None, None, None, :]
        #
        # ww_range = self.list_parameter_b[3] # [a]
        # ww_range = ww_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_dim, 1).view(-1, self.out_channels)  # (batch, 200)
        # ww_range = ww_range[:, :, None, None, None]
        #
        # f_out = f_out + xx_range + yy_range + zz_range + ww_range

        if self.norm is not None:
            f_out = self.norm(f_out)

        f_out = self.act_func(f_out)
        return f_out

class LRLC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1,act_func='leaky', norm_layer='bn', bias=False, np_feature_map_size = None, n_K = 1):
        super(LRLC, self).__init__()
        self.n_K = n_K

        self.cnn_layers = nn.ModuleList([])
        for i in range(n_K):
            self.cnn_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias))

        if norm_layer == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm_layer is None:
            self.norm = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'leaky':
            self.act_func = nn.LeakyReLU()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

        #TODO : define the parameters
        self.out_channels = out_channels # out_plane
        self.shape_feature_map = np_feature_map_size # [w, h, d]
        self.dim_feature_map = np_feature_map_size.shape[0] # 3

        """ one initialized weight vectors for width, height, depth"""
        self.list_K = nn.ParameterList([])
        for i in range(self.dim_feature_map):
            alpha = nn.Parameter(torch.ones(self.n_K, np_feature_map_size[i]))
            nn.init.uniform_(alpha, a=0.0, b=1.0)
            self.list_K.append(alpha)

        """ zero initialized bias vectors for width, height, depth"""
        self.list_parameter_b = nn.ParameterList([])
        for i in range(self.dim_feature_map):
            alpha = nn.Parameter(torch.zeros(np_feature_map_size[i]))
            self.list_parameter_b.append(alpha)
        alpha = nn.Parameter(torch.zeros(out_channels))
        self.list_parameter_b.append(alpha)

    def forward(self, input_tensor):
        """ weight """
        f_out = torch.zeros(list(input_tensor.shape[:1]) + [self.out_channels] + list(self.shape_feature_map), dtype=torch.float32).cuda()
        for i in range(self.n_K):
            out = self.cnn_layers[i](input_tensor)
            w_dim = out.shape[1]  # 44
            x_dim = out.shape[2]  # 44
            y_dim = out.shape[3]  # 54
            z_dim = out.shape[4]  # 41
            batch_size_tensor = out.shape[0]
            xx_ones = torch.ones([batch_size_tensor, z_dim], dtype=torch.float32).cuda()
            xx_ones = xx_ones[:, :, None]  # batch, z_dim, 1 (6, 41, 1)
            xx_range = self.list_K[0][i]
            xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, x_dim)
            xx_range = xx_range[:, None, :] # batch, 1, x_dim
            xx_channel = torch.matmul(xx_ones, xx_range) # batch, z_dim, x_dim
            xx_channel = xx_channel.unsqueeze(3).repeat(1, 1, 1, y_dim).unsqueeze(1).float() # batch, 1, z_dim, x_dim, y_dim
            xx_channel = xx_channel.permute(0, 1, 3, 4, 2) # batch, 1, x, y, z

            yy_ones = torch.ones([batch_size_tensor, x_dim], dtype=torch.float32).cuda()
            yy_ones = yy_ones[:, :, None]  # (batch, 175, 1)
            yy_range = self.list_K[1][i]
            yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, y_dim)  # (batch, 200)
            yy_range = yy_range[:, None, :]
            yy_channel = torch.matmul(yy_ones, yy_range)
            yy_channel = yy_channel.unsqueeze(3).repeat(1, 1, 1, z_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
            yy_channel = yy_channel.permute(0, 1, 2, 3, 4)

            zz_ones = torch.ones([batch_size_tensor, y_dim], dtype=torch.float32).cuda()
            zz_ones = zz_ones[:, :, None]  # (batch, 175, 1)
            zz_range = self.list_K[2][i]
            zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, z_dim)  # (batch, 200)
            zz_range = zz_range[:, None, :]
            zz_channel = torch.matmul(zz_ones, zz_range)
            zz_channel = zz_channel.unsqueeze(3).repeat(1, 1, 1, x_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
            zz_channel = zz_channel.permute(0, 1, 4, 2, 3)

            ## TODO : normalize w matrix
            large_w = (xx_channel + yy_channel + zz_channel)
            # large_w = nn.Softmax(-1)(large_w.contiguous().view(large_w.size()[0], large_w.size()[1], -1)).view_as(large_w)
            large_w = nn.Sigmoid()(large_w)
            f_out += large_w * out

        """ bias """
        xx_range = self.list_parameter_b[0]
        xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, x_dim)
        xx_range = xx_range[:, None, :, None, None]

        yy_range = self.list_parameter_b[1]
        yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, y_dim)  # (batch, 200)
        yy_range = yy_range[:, None, None, :, None]

        zz_range = self.list_parameter_b[2]
        zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, z_dim)  # (batch, 200)
        zz_range = zz_range[:, None, None, None, :]

        ww_range = self.list_parameter_b[3] # [a]
        ww_range = ww_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, w_dim)  # (batch, 200)
        ww_range = ww_range[:, :, None, None, None]

        f_out = f_out + xx_range + yy_range + zz_range + ww_range

        if self.norm is not None:
            f_out = self.norm(f_out)
        if self.act_func is not None:
            f_out = self.act_func(f_out)
        return f_out

class LRLC_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1,act_func='leaky', norm_layer='bn', bias=False, np_feature_map_size = None, n_K = 1):
        super(LRLC_2, self).__init__()
        self.n_K = n_K

        self.cnn_layers = nn.ModuleList([])
        for i in range(n_K):
            self.cnn_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias))

        if norm_layer == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm_layer is None:
            self.norm = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'leaky':
            self.act_func = nn.LeakyReLU()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

        #TODO : define the parameters
        self.out_channels = out_channels # out_plane
        self.shape_feature_map = np_feature_map_size # [w, h, d]
        self.dim_feature_map = np_feature_map_size.shape[0] # 3

        """ one initialized weight vectors for width, height, depth"""
        self.list_K = nn.ParameterList([])
        for i in range(self.n_K):
            alpha = nn.Parameter(torch.ones(tuple(self.shape_feature_map)))
            nn.init.uniform_(alpha, a=0.0, b=1.0)
            self.list_K.append(alpha)


        """ zero initialized bias vectors for width, height, depth"""
        # self.list_parameter_b = nn.ParameterList([])
        # for i in range(self.dim_feature_map):
        #     alpha = nn.Parameter(torch.zeros(np_feature_map_size[i]))
        #     # nn.init.uniform_(alpha, a=0.0, b=1.0)
        #     self.list_parameter_b.append(alpha)
        # alpha = nn.Parameter(torch.zeros(out_channels))
        # self.list_parameter_b.append(alpha)

    def forward(self, input_tensor):
        """ weight """
        f_out = torch.zeros(list(input_tensor.shape[:1]) + [self.out_channels] + list(self.shape_feature_map), dtype=torch.float32).cuda()
        for i in range(self.n_K):
            out = self.cnn_layers[i](input_tensor)
            large_w = self.list_K[i]
            ## TODO : normalize w matrix
            # large_w = nn.Softmax(-1)(large_w.contiguous().view(large_w.size()[0], large_w.size()[1], -1)).view_as(large_w)
            # large_w = nn.Sigmoid()(large_w)
            f_out += large_w * out

        if self.norm is not None:
            f_out = self.norm(f_out)
        if self.act_func is not None:
            f_out = self.act_func(f_out)
        return f_out

class LRLC_RoI(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups = 1,act_func='leaky', norm_layer='bn', bias=False, np_feature_map_size = None, n_K = 1, n_RoI = None, RoI_template = None):
        super(LRLC_RoI, self).__init__()
        self.n_K = n_K
        self.n_RoI = n_RoI
        self.RoI_template = RoI_template
        self.cnn_layers = nn.ModuleList([])
        for i in range(n_K):
            self.cnn_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias))

        if norm_layer == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm_layer is None:
            self.norm = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'leaky':
            self.act_func = nn.LeakyReLU()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

        #TODO : define the parameters
        self.out_channels = out_channels # out_plane
        self.shape_feature_map = np_feature_map_size # [w, h, d]
        self.dim_feature_map = np_feature_map_size.shape[0] # 3

        """ one initialized weight vectors for width, height, depth"""
        self.list_K = nn.ParameterList([])
        for i in range(self.n_K):
            alpha = nn.Parameter(torch.ones(self.n_RoI))
            nn.init.uniform_(alpha, a=0.0, b=1.0)
            self.list_K.append(alpha)

        """ zero initialized bias vectors for width, height, depth"""
        self.param_b = nn.Parameter(torch.zeros(self.n_RoI))

    def forward(self, input_tensor):
        """ weight """
        f_out = torch.zeros(list(input_tensor.shape[:1]) + [self.out_channels] + list(self.shape_feature_map), dtype=torch.float32).cuda()
        for i in range(self.n_K):
            out = self.cnn_layers[i](input_tensor)
            large_w = self.list_K[i]

            ## TODO : normalize w matrix
            large_w = nn.Softmax(-1)(large_w)
            for j in range(self.n_RoI):
                out[:, :, (self.RoI_template.squeeze() == j + 1)] *= large_w[j]
            f_out += out

        for j in range(self.n_RoI):
            f_out[:, :, (self.RoI_template.squeeze() == j + 1)] += self.param_b[j]

        if self.norm is not None:
            f_out = self.norm(f_out)
        if self.act_func is not None:
            f_out = self.act_func(f_out)
        return f_out

class BasicConv_Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, act_func='leaky', norm_layer='in', bias=False):
        super(BasicConv_Block, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate', dilation=dilation, groups=groups, bias=bias)
        # self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate', dilation=dilation, groups=groups, bias=bias)

        if norm_layer == 'bn':
            self.norm = nn.BatchNorm3d(out_planes)
        elif norm_layer == 'in':
            # self.norm = nn.InstanceNorm3d(out_planes, affine=True)
            self.norm = nn.InstanceNorm3d(out_planes, affine=True)
        elif norm_layer == 'layerNorm':
            self.norm = LayerNorm3d(out_planes)
        elif norm_layer is None:
            self.norm = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'leaky':
            self.act_func = nn.LeakyReLU()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act_func is not None:
            x = self.act_func(x)

        return x

class BasicConv_Block_1D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, act_func='leaky', norm_layer='in', bias=False):
        super(BasicConv_Block_1D, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate',dilation=dilation, groups=groups, bias=bias)

        if norm_layer == 'bn':
            self.norm = nn.BatchNorm1d(out_planes)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm1d(out_planes, affine=False)
        elif norm_layer == 'layerNorm':
            self.norm = LayerNorm3d(out_planes)
        elif norm_layer is None:
            self.norm = None
        else:
            assert False

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_func == 'leaky':
            self.act_func = nn.LeakyReLU()
        elif act_func is None:
            self.act_func = None
        else:
            assert False

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x

class Basic_residual_block(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, dilation=1, padding = 0):
        super(Basic_residual_block, self).__init__()
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate', bias=False)
        self.norm1 = nn.InstanceNorm3d(planes, affine=False)
        # self.norm1 = nn.BatchNorm3d(planes, affine=True)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='replicate', bias=False)
        self.norm2 = nn.InstanceNorm3d(planes, affine=False)
        # self.norm2 = nn.BatchNorm3d(planes, affine=True)
        self.act_f = nn.ReLU(inplace=True)
        # self.act_f = nn.Tanh()
        # self.act_f = nn.LeakyReLU(inplace=True)

        if stride != 1 or inplanes != planes * self.expansion :
            self.downsample = nn.Sequential(
                    nn.Conv3d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.InstanceNorm3d(planes * self.expansion, affine=False),
                    # nn.BatchNorm3d(planes * self.expansion, affine=True),
                )
        else :
            self.downsample = None

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_f(out)

        out = self.conv2(out)
        out = self.norm2(out)

        ## Adjust decreasing factor by kernel size
        if self.padding == 0 :
            residual = x[:, :, self.kernel_size // 2 + (self.dilation - 1):, self.kernel_size // 2 + (self.dilation - 1):, self.kernel_size // 2 + (self.dilation - 1):]
        else:
            residual = x

        ## downsample the residual
        if self.downsample is not None:
            residual = self.downsample(residual)

        ## crop and residual connection
        out += residual[:, :, :out.size()[-3], :out.size()[-2], :out.size()[-1]]

        out = self.act_f(out)
        return out

class Basic_residual_block_v2(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, dilation=1, padding = 0, flag_res = True):
        super(Basic_residual_block_v2, self).__init__()
        self.padding = padding
        self.dilation = dilation
        self.flag_res = flag_res

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate', bias=False)
        self.norm1 = nn.InstanceNorm3d(planes, affine=True)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=1, padding=padding, padding_mode='replicate', bias=False)
        self.norm2 = nn.InstanceNorm3d(planes, affine=True)

        self.act_f = nn.ReLU(inplace=True)

        if self.flag_res == True:
            if stride != 1 or inplanes != planes * self.expansion :
                self.downsample = nn.Sequential(
                        nn.Conv3d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                        nn.InstanceNorm3d(planes * self.expansion, affine=True),
                    )
            else :
                self.downsample = None

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_f(out)

        out = self.conv2(out)
        out = self.norm2(out)

        ## Adjust decreasing factor by kernel size
        if self.padding == 0 :
            residual = x[:, :, self.kernel_size // 2 + (self.dilation - 1):, self.kernel_size // 2 + (self.dilation - 1):, self.kernel_size // 2 + (self.dilation - 1):]
        else:
            residual = x

        if self.flag_res == True:
            ## downsample the residual
            if self.downsample is not None:
                residual = self.downsample(residual)

            ## crop and residual connection
            out += residual[:, :, :out.size()[-3], :out.size()[-2], :out.size()[-1]]

        out = self.act_f(out)
        return out

class Basic_residual_block_v3(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, dilation=1, padding = 0, flag_res = True):
        super(Basic_residual_block_v3, self).__init__()
        self.padding = padding
        self.dilation = dilation
        self.flag_res = flag_res
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='replicate', bias=False)
        self.norm1 = nn.InstanceNorm3d(planes, affine=False)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=1, stride=1, padding=0, padding_mode='replicate', bias=False)
        self.norm2 = nn.InstanceNorm3d(planes, affine=False)

        self.act_f = nn.ReLU(inplace=True)

        if self.flag_res == True:
            if stride != 1 or inplanes != planes * self.expansion :
                self.downsample = nn.Sequential(
                        nn.Conv3d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                        nn.InstanceNorm3d(planes * self.expansion, affine=False),
                    )
            else :
                self.downsample = None

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_f(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.flag_res == True:
            ## downsample the residual
            if self.downsample is not None:
                residual = self.downsample(residual)

            ## crop and residual connection
            out += residual[:, :, :out.size()[-3], :out.size()[-2], :out.size()[-1]]

        out = self.act_f(out)
        return out

class GateNet(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_r=False, mode='both'):
        super(GateNet, self).__init__()
        if with_r == False:
            self.inplane = 3
        else :
            self.inplane = 4

        assert mode =='coord' or mode == 'feature'
        self.mode = mode
        mid_planes = in_planes // 2

        if mode == 'coord':
            self.coordEmbedding = nn.Sequential(
                nn.Conv3d(self.inplane, mid_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
                nn.InstanceNorm3d(mid_planes, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
                nn.InstanceNorm3d(in_planes, affine=False),
                nn.ReLU(inplace=True),
            )
            self.gate = nn.Sequential(
                nn.Conv3d(in_planes, mid_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
                nn.InstanceNorm3d(mid_planes, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_planes, out_planes, kernel_size=1, stride=1, bias=False),
            )

        elif mode == 'feature':
            self.gate = nn.Sequential(
                nn.Conv3d(in_planes, mid_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
                nn.InstanceNorm3d(mid_planes, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_planes, out_planes, kernel_size=1, stride=1, bias=False),
            )


    def forward(self, x, coord):
        if self.mode == 'coord':
            gate = self.coordEmbedding(coord)
        elif self.mode == 'feature':
            gate = x
        gate = self.gate(gate)
        gate = torch.mean(gate, dim=1, keepdim=True)
        gate = torch.sigmoid(gate)
        return gate

def calcu_featureMap_dim(input_size, kernel, stride, padding, dilation):
    padding = np.tile(padding, len(input_size))
    kernel = np.tile(kernel, len(input_size))
    stride = np.tile(stride, len(input_size))
    dilation = np.tile(dilation, len(input_size))

    t_inputsize = np.array(input_size) + (padding * 2)
    t_kernel = (kernel-1) * dilation + 1
    output_size = (t_inputsize - t_kernel) // stride + 1
    return output_size

class AddCoords_size(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoords_size, self).__init__()

        self.with_r = with_r
    def forward(self, size):
        # size = [xdim, y_dim, z_dim]
        x_dim = size[0]
        y_dim = size[1]
        z_dim = size[2]

        xx_ones = torch.ones([z_dim], dtype=torch.float32).cuda()
        xx_ones = xx_ones[:, None]  # (175, 1)
        xx_range = torch.arange(0, x_dim, dtype=torch.float32).cuda()  # (200,)
        xx_range = xx_range[None, :]  # (1, 200)
        xx_channel = torch.matmul(xx_ones, xx_range) # (175, 200)
        xx_channel = xx_channel.unsqueeze(-1).repeat(1, 1, y_dim).float() # (175, 200, 143)
        xx_channel /= (x_dim - 1)
        xx_channel = xx_channel * 2 - 1
        xx_channel = xx_channel.permute(1, 2, 0).unsqueeze(0)
        del xx_ones, xx_range

        yy_ones = torch.ones([x_dim], dtype=torch.float32).cuda()
        yy_ones = yy_ones[:, None]  # (batch, 175, 1)
        yy_range = torch.arange(0, y_dim, dtype=torch.float32).cuda()  # (200,)
        yy_range = yy_range[None, :]  # (batch, 1, 200)
        yy_channel = torch.matmul(yy_ones, yy_range) # (4, 175, 200)
        yy_channel = yy_channel.unsqueeze(-1).repeat(1, 1, z_dim).float() # (4, 1, 175, 200, 143)
        yy_channel /= (y_dim - 1)
        yy_channel = yy_channel * 2 - 1
        yy_channel = yy_channel.permute(0, 1, 2).unsqueeze(0)
        del yy_ones, yy_range

        zz_ones = torch.ones([y_dim], dtype=torch.float32).cuda()
        zz_ones = zz_ones[:, None]  # (batch, 175, 1)
        zz_range = torch.arange(0, z_dim, dtype=torch.float32).cuda()  # (200,)
        zz_range = zz_range[None, :]  # (batch, 1, 200)
        zz_channel = torch.matmul(zz_ones, zz_range) # (4, 175, 200)
        zz_channel = zz_channel.unsqueeze(-1).repeat(1, 1, x_dim).float() # (4, 1, 175, 200, 143)
        zz_channel /= (z_dim - 1)
        zz_channel = zz_channel * 2 - 1
        zz_channel = zz_channel.permute(2, 0, 1).unsqueeze(0)
        del zz_ones, zz_range

        ret = torch.cat([xx_channel, yy_channel, zz_channel], 0).unsqueeze(0)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2) + torch.pow(zz_channel, 2)).unsqueeze(0)
            ret = torch.cat([ret, rr], dim=1)
        return ret

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()

        self.with_r = with_r
    def forward(self, input_tensor):
        # batch, 1, x, y, z
        x_dim = input_tensor.shape[2]
        y_dim = input_tensor.shape[3]
        z_dim = input_tensor.shape[4]
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([batch_size_tensor, z_dim], dtype=torch.float32).cuda()
        xx_ones = xx_ones[:, :, None]  # (batch, 175, 1)
        xx_range = torch.arange(0, x_dim, dtype=torch.float32).cuda()  # (200,)
        xx_range = xx_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, x_dim)  # (batch, 200)
        xx_range = xx_range[:, None, :]  # (batch, 1, 200)
        xx_channel = torch.matmul(xx_ones, xx_range) # (4, 175, 200)
        xx_channel = xx_channel.unsqueeze(3).repeat(1, 1, 1, y_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
        del xx_ones, xx_range
        xx_channel /= (x_dim - 1)
        xx_channel = xx_channel * 2 - 1
        xx_channel = xx_channel.permute(0,1,3,4,2)

        yy_ones = torch.ones([batch_size_tensor, x_dim], dtype=torch.float32).cuda()
        yy_ones = yy_ones[:, :, None]  # (batch, 175, 1)
        yy_range = torch.arange(0, y_dim, dtype=torch.float32).cuda()  # (200,)
        yy_range = yy_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, y_dim)  # (batch, 200)
        yy_range = yy_range[:, None, :]  # (batch, 1, 200)
        yy_channel = torch.matmul(yy_ones, yy_range) # (4, 175, 200)
        yy_channel = yy_channel.unsqueeze(3).repeat(1, 1, 1, z_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
        del yy_ones, yy_range
        yy_channel /= (y_dim - 1)
        yy_channel = yy_channel * 2 - 1
        yy_channel = yy_channel.permute(0, 1, 2, 3, 4)

        zz_ones = torch.ones([batch_size_tensor, y_dim], dtype=torch.float32).cuda()
        zz_ones = zz_ones[:, :, None]  # (batch, 175, 1)
        zz_range = torch.arange(0, z_dim, dtype=torch.float32).cuda()  # (200,)
        zz_range = zz_range.unsqueeze(0).transpose(0, 1).repeat(1, batch_size_tensor, 1).view(-1, z_dim)  # (batch, 200)
        zz_range = zz_range[:, None, :]  # (batch, 1, 200)
        zz_channel = torch.matmul(zz_ones, zz_range) # (4, 175, 200)
        zz_channel = zz_channel.unsqueeze(3).repeat(1, 1, 1, x_dim).unsqueeze(1).float() # (4, 1, 175, 200, 143)
        del zz_ones, zz_range
        zz_channel /= (z_dim - 1)
        zz_channel = zz_channel * 2 - 1
        zz_channel = zz_channel.permute(0, 1, 4, 2, 3)
        ret = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], 1)
        # ret = torch.cat([xx_channel, yy_channel, zz_channel], 1)
        return ret

class CoordConv(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1 , groups = 1, bias=False, with_r = False):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        self.conv = nn.Conv3d(in_channels+3, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act_f = nn.LeakyReLU(inplace=True)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        ret = self.norm(ret)
        ret = self.act_f(ret)
        return ret



class sign_sqrt(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.sign(input) * torch.sqrt(torch.abs(input))
        # output = torch.sqrt(input.abs())
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        grad_input = torch.div(grad_output, ((torch.abs(output)+0.03)*2.))
        return grad_input

class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m

class XceptionConv_layer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(XceptionConv_layer, self).__init__()
        self.out_channels = out_planes
        self.conv = SeparableConv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Multi_Heads_Self_Attn_1D(nn.Module):
    def __init__(self, in_plane, n_heads = 4, d_k = 16):
        super(Multi_Heads_Self_Attn_1D, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv1d(in_plane, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=True)
        self.key_conv = nn.Conv1d(in_plane, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=True)
        self.value_conv = nn.Conv1d(in_plane, self.num_heads * self.d_k, kernel_size=1, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv1d(self.num_heads * self.d_k, in_plane, kernel_size=1, bias=True)
        self.act_f = nn.ReLU(inplace=True)
        self.norm = nn.InstanceNorm1d(in_plane, affine=True)

        """ gamma """
        self.gamma = nn.Parameter(torch.zeros(1))
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x):
        m_batchsize, C, depth= x.size()
        total_key_depth = depth

        """ linear for each component"""
        queries = self.query_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        keys = self.key_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        values = self.value_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ query scale"""
        query_scale = np.power(self.d_k, -0.5)
        queries *= query_scale
        logits = torch.matmul(queries, keys.permute(0,1,3,2))
        weights = self.softmax(logits)  # BX (N) X (N/p)
        out = torch.matmul(weights, values)

        # logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        # weights = logits / self.d_k
        # out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, depth, -1).permute(0, 2, 1)
        out = self.output_conv(out)
        # out = self.norm(out)
        # out = self.drop_1(out)
        # out = out * self.gamma + x
        out = out + x
        out = self.act_f(out)
        return out

class PositionWiseFeedForward(nn.Module):
    def __init__(self, n_featuremap):
        super().__init__()
        self.ff_1 = nn.Conv3d(n_featuremap, n_featuremap * 2, kernel_size=1, bias=False)
        self.act_f = nn.ReLU(inplace=True)
        self.ff_2 = nn.Conv3d(n_featuremap * 2, n_featuremap, kernel_size=1, bias=False)
        self.norm = LayerNorm3d(n_featuremap)
        self.drop_2 = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x
        out = self.ff_1(x)
        out = self.act_f(out)
        out = self.ff_2(out)
        out = self.norm(out)
        out = self.act_f(out)
        out = self.drop_2(out)
        out = residual + out
        return out


class Multi_Heads_Self_Attn(nn.Module):
    def __init__(self, n_featuremap, n_heads = 4,  d_k = 16):
        super(Multi_Heads_Self_Attn, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv3d(n_featuremap, self.num_heads * self.d_k , kernel_size=1, padding=0, bias=False)

        """ key """
        self.key_conv = nn.Conv3d(n_featuremap, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)

        """ value """
        self.value_conv = nn.Conv3d(n_featuremap, self.num_heads * self.d_k, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv3d(self.num_heads * self.d_k, n_featuremap, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(n_featuremap)
        self.act_f = nn.ReLU(inplace=True)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x):
        m_batchsize, C, width, height, depth = x.size()
        total_key_depth = width * height * depth

        """ linear for each component"""
        queries = self.query_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        keys = self.key_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)
        values = self.value_conv(x).view(m_batchsize, -1, total_key_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys )
        values = self._split_heads(values)


        """ query scale"""
        query_scale = np.power(self.d_k, -0.5)
        queries *= query_scale
        logits = torch.matmul(queries, keys.permute(0,1,3,2))
        weights = self.softmax(logits)  # BX (N) X (N/p)
        out = torch.matmul(weights, values)

        """Combine queries and keys """
        # logits = torch.matmul(queries, keys.permute(0,1,3,2))
        # weights = logits / logits.size(-1)
        # out = torch.matmul(weights, values)


        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, width, height, depth, -1).permute(0, 4, 1, 2, 3)
        out = self.output_conv(out)
        out = self.norm(out)

        """ residual """
        out = out + x
        out = self.act_f(out)

        return out

class Multi_Heads_Self_Attn_Q_KV(nn.Module):
    def __init__(self, n_q, n_kv, n_heads = 4, d_k = 16):
        super(Multi_Heads_Self_Attn_Q_KV, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv3d(n_q, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)
        self.key_conv = nn.Conv3d(n_kv, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)
        self.value_conv = nn.Conv3d(n_kv, self.num_heads * self.d_k, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv3d(self.num_heads * self.d_k, n_q, kernel_size=1, bias=False)
        self.norm_layer_q = nn.InstanceNorm3d(n_q, affine=True)
        self.norm_layer_k = nn.InstanceNorm3d(n_q, affine=True)
        self.norm_layer_pwff = nn.InstanceNorm3d(n_q, affine=True)
        # self.norm_layer = LayerNorm3d(n_q)
        self.act_f = nn.ReLU(inplace=True)

        self.pwff = nn.Sequential(
            nn.Conv3d(n_q, n_q * 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_q * 2, n_q, kernel_size=1, bias=True),
        )

        """ gamma """
        # self.gamma = nn.Parameter(torch.zeros(1))

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x_q, x_kv, mask):
        m_batchsize, C, width, height, depth = x_q.size()
        total_q_depth = width * height * depth

        m_batchsize, C, width_kv, height_kv, depth_kv = x_kv.size()
        total_kv_depth = width_kv * height_kv * depth_kv

        """ linear for each component"""
        x_q_2 = self.norm_layer_q(x_q)
        x_kv_2 = self.norm_layer_k(x_kv)
        queries = self.query_conv(x_q_2).view(m_batchsize, -1, total_q_depth).permute(0, 2, 1)
        keys = self.key_conv(x_kv_2).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)
        values = self.value_conv(x_kv_2).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ query scale"""
        # query_scale = np.power(self.d_k, -0.5)
        # queries = queries * query_scale


        logits = torch.matmul(queries, keys.permute(0,1,3,2)) # [batch, head, 1, length]


        # weights = self.softmax(logits)  # BX (N) X (N/p)
        weights = torch.sigmoid(logits)
        if mask is not None:
            mask = mask.view(mask.size(0), mask.size(1), -1)
            # logits -= ((1.0 - mask.unsqueeze(-2)) * 10000)
            # logits -= (torch.log(1.0 - mask.unsqueeze(-2) + 1e-7))
            weights = weights * mask.unsqueeze(-2)
        # weights = weights / weights.sum(dim=-1, keepdim=True)


        out = torch.matmul(weights, values) # [batch, head, 1, length]


        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, width, height, depth, -1).permute(0, 4, 1, 2, 3)
        out = self.output_conv(out) + x_q
        # out = self.output_conv(out) * self.gamma + x_q

        out_res = out
        out = self.norm_layer_pwff(out)
        out = self.pwff(out) + out_res

        return out

class Multi_Heads_Self_Attn_Q_KV_dot_prod(nn.Module):
    def __init__(self, n_q, n_kv, n_heads = 4, d_k = 16):
        super(Multi_Heads_Self_Attn_Q_KV_dot_prod, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv3d(n_q, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)
        self.key_conv = nn.Conv3d(n_kv, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=False)
        self.value_conv = nn.Conv3d(n_kv, self.num_heads * self.d_k, kernel_size=1, bias=False)

        self.output_conv = nn.Conv3d(self.num_heads * self.d_k, n_q, kernel_size=1, bias=False)
        self.act_f = nn.ReLU(inplace=True)

        """ gamma """
        self.gamma = nn.Parameter(torch.zeros(1))

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x_q, x_kv):
        m_batchsize, C, width, height, depth = x_q.size()
        total_q_depth = width * height * depth

        m_batchsize, C, width_kv, height_kv, depth_kv = x_kv.size()
        total_kv_depth = width_kv * height_kv * depth_kv

        """ linear for each component"""
        queries = self.query_conv(x_q).view(m_batchsize, -1, total_q_depth).permute(0, 2, 1)
        keys = self.key_conv(x_kv).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)
        values = self.value_conv(x_kv).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ query scale"""
        logits = torch.matmul(queries, keys.permute(0,1,3,2)) # [batch, head, 1, length]
        N = logits.size(-1)
        weights = logits / N
        out = torch.matmul(weights, values) # [batch, head, 1, length]

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, width, height, depth, -1).permute(0, 4, 1, 2, 3)
        out = self.output_conv(out)
        # out = self.norm_layer(out)

        """ residual """
        # out = self.gamma * out
        out = out + x_q
        out = self.act_f(out)
        return out

class Multi_Heads_Self_Attn_Q_KV_1D(nn.Module):
    def __init__(self, n_featuremap_q, n_featuremap_kv, n_heads = 4, d_k = 16):
        super(Multi_Heads_Self_Attn_Q_KV_1D, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        """ query key value """
        self.query_conv = nn.Conv1d(n_featuremap_q, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=True)
        self.key_conv = nn.Conv1d(n_featuremap_kv, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=True)
        self.value_conv = nn.Conv1d(n_featuremap_kv, self.num_heads * self.d_k, kernel_size=1, bias=True)

        self.coord_q_conv = nn.Conv1d(3, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=True)
        self.coord_k_conv = nn.Conv1d(3, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv1d(self.num_heads * self.d_k, n_featuremap_q, kernel_size=1, bias=True)
        self.norm = nn.InstanceNorm1d(n_featuremap_q, affine=True)
        self.act_f = nn.ReLU(inplace=True)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x_q, x_kv, coord_q, coord_k):
        m_batchsize, C, width = x_q.size()
        total_q_depth = width

        m_batchsize, C, width_kv = x_kv.size()
        total_kv_depth = width_kv

        """ linear for each component"""
        queries = self.query_conv(x_q).view(m_batchsize, -1, total_q_depth).permute(0, 2, 1)
        keys = self.key_conv(x_kv).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)
        values = self.value_conv(x_kv).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)    # B X N X Channel

        coord_q = self.coord_q_conv(coord_q).view(m_batchsize, -1, total_q_depth).permute(0, 2, 1)
        coord_k = self.coord_k_conv(coord_k).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)
        queries += coord_q
        keys += coord_k

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ query scale"""
        query_scale = np.power(self.d_k, -0.5)
        queries *= query_scale
        logits = torch.matmul(queries, keys.permute(0,1,3,2))
        weights = self.softmax(logits)  # BX (N) X (N/p)

        """Combine queries and keys """
        # logits = torch.matmul(queries, keys.permute(0,1,3,2))
        # weights = logits / self.d_k

        out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, width, -1).permute(0, 2, 1)
        out = self.output_conv(out)
        out = self.norm(out)

        """ residual """
        out = out + x_q
        out = self.act_f(out)
        return out



class VariationalPosterior(torch.nn.Module):
    def __init__(self, mu, rho):
        super(VariationalPosterior, self).__init__()
        self.mu = mu.cuda()
        self.rho = rho.cuda()
        # gaussian distribution to sample epsilon from
        self.normal = torch.distributions.Normal(0, 1)
        self.sigma = torch.log1p(torch.exp(self.rho)).cuda()

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).cuda()
        # reparametrizarion trick for sampling from posterior
        posterior_sample = (self.mu + self.sigma * epsilon).cuda()
        return posterior_sample

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class Prior(torch.nn.Module):
    '''
    Scaled Gaussian Mixtures for Priors
    '''
    def __init__(self, args):
        super(Prior, self).__init__()
        self.sig1 = args.sig1
        self.sig2 = args.sig2
        self.pi = args.pi


        self.s1 = torch.tensor([math.exp(-1. * self.sig1)], dtype=torch.float32).cuda()
        self.s2 = torch.tensor([math.exp(-1. * self.sig2)], dtype=torch.float32).cuda()

        self.gaussian1 = torch.distributions.Normal(0,self.s1)
        self.gaussian2 = torch.distributions.Normal(0,self.s2)


    def log_prob(self, input):
        input = input.cuda()
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1.-self.pi) * prob2)).sum()


class MS_GRU(nn.Module):
    def __init__(self, in_planes):
        super(MS_GRU, self).__init__()
        self.W_z = nn.Conv3d(in_planes * 2, in_planes, kernel_size=1, padding=0, bias=True)
        self.W_r = nn.Conv3d(in_planes * 2, in_planes, kernel_size=1, padding=0, bias=True)
        self.W_h = nn.Conv3d(in_planes * 2, in_planes, kernel_size=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, h_t_1, x_t):
        z_t = self.sigmoid(self.W_z(torch.cat([h_t_1, x_t], dim=1)))
        r_t = self.sigmoid(self.W_r(torch.cat([h_t_1, x_t], dim=1)))
        h_t_til = self.tanh(self.W_h(torch.cat([r_t * h_t_1, x_t], dim=1)))
        h_t = (1 - z_t) * h_t_1 + z_t * h_t_til
        return h_t

class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input

class gdl_loss(nn.Module):
    def __init__(self, pNorm=2):
        super(gdl_loss, self).__init__()
        self.convX = nn.Conv3d(1, 1, kernel_size=(1, 1, 2), stride=1, padding=(0, 0, 1), bias=False)
        self.convY = nn.Conv3d(1, 1, kernel_size=(1, 2, 1), stride=1, padding=(0, 1, 0), bias=False)
        self.convZ = nn.Conv3d(1, 1, kernel_size=(2, 1, 1), stride=1, padding=(1, 0, 0), bias=False)

        filterX = torch.FloatTensor([[[[[1, -1]]]]])  # 1x1x2
        filterY = torch.FloatTensor([[[[[1], [-1]]]]])  # 1x2x1
        filterZ = torch.FloatTensor([[[[[1]], [[-1]]]]])  # 2x1x1

        self.convX.weight = torch.nn.Parameter(filterX, requires_grad=False)
        self.convY.weight = torch.nn.Parameter(filterY, requires_grad=False)
        self.convZ.weight = torch.nn.Parameter(filterZ, requires_grad=False)
        self.pNorm = pNorm

    def forward(self, pred, gt):
        assert not gt.requires_grad
        assert pred.dim() == 5
        assert gt.dim() == 5
        assert pred.size() == gt.size(), "{0} vs {1} ".format(pred.size(), gt.size())
        pred_dx = torch.abs(self.convX(pred))
        pred_dy = torch.abs(self.convY(pred))
        pred_dz = torch.abs(self.convZ(pred))

        gt_dx = torch.abs(self.convX(gt))
        gt_dy = torch.abs(self.convY(gt))
        gt_dz = torch.abs(self.convZ(gt))

        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)
        grad_diff_z = torch.abs(gt_dz - pred_dz)

        mat_loss_x = grad_diff_x ** self.pNorm
        mat_loss_y = grad_diff_y ** self.pNorm  # Batch x Channel x width x height
        mat_loss_z = grad_diff_z ** self.pNorm

        shape = gt.shape
        # mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y) + torch.sum(mat_loss_z)) / (
        #             shape[0] * shape[1] * shape[2] * shape[3] * shape[4])
        mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y) + torch.sum(mat_loss_z))
        return mean_loss


class ReplayBuffer():
    def __init__(self, max_size=10):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class gdl_loss(nn.Module):
    def __init__(self, pNorm=1):
        super(gdl_loss, self).__init__()
        self.convX = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=(0, 1), bias=False)
        self.convY = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=(1, 0), bias=False)

        filterX = torch.FloatTensor([[[[1, -1]]]]).cuda()  # 1x2
        filterY = torch.FloatTensor([[[[1], [-1]]]]).cuda()  # 1x2x1

        self.convX.weight = torch.nn.Parameter(filterX, requires_grad=False)
        self.convY.weight = torch.nn.Parameter(filterY, requires_grad=False)
        self.pNorm = pNorm

    def forward(self, pred, gt):
        assert not gt.requires_grad
        assert pred.dim() == 4
        assert gt.dim() == 4
        assert pred.size() == gt.size(), "{0} vs {1} ".format(pred.size(), gt.size())
        pred_dx = torch.abs(self.convX(pred))
        pred_dy = torch.abs(self.convY(pred))

        gt_dx = torch.abs(self.convX(gt))
        gt_dy = torch.abs(self.convY(gt))

        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)

        mat_loss_x = grad_diff_x ** self.pNorm
        mat_loss_y = grad_diff_y ** self.pNorm  # Batch x Channel x width x height

        shape = gt.shape
        mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y) /
                     (shape[0] * shape[1] * shape[2] * shape[3]))
        # mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y))
        return mean_loss

class gdl_loss_2(nn.Module):
    def __init__(self, pNorm=1):
        super(gdl_loss_2, self).__init__()
        self.convX = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=(0, 1), bias=False)
        self.convY = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=(1, 0), bias=False)

        filterX = torch.FloatTensor([[[[1, -1]]]]).cuda()  # 1x2
        filterY = torch.FloatTensor([[[[1], [-1]]]]).cuda()  # 1x2x1

        self.convX.weight = torch.nn.Parameter(filterX, requires_grad=False)
        self.convY.weight = torch.nn.Parameter(filterY, requires_grad=False)
        self.pNorm = pNorm

    def forward(self, pred):
        assert pred.dim() == 5
        gt = torch.zeros_like(pred)
        pred_dx = torch.abs(self.convX(pred))
        pred_dy = torch.abs(self.convY(pred))

        gt_dx = torch.abs(self.convX(gt))
        gt_dy = torch.abs(self.convY(gt))

        grad_diff_x = torch.abs(gt_dx - pred_dx)
        grad_diff_y = torch.abs(gt_dy - pred_dy)

        mat_loss_x = grad_diff_x ** self.pNorm
        mat_loss_y = grad_diff_y ** self.pNorm  # Batch x Channel x width x height

        shape = gt.shape
        mean_loss = ((torch.sum(mat_loss_x) + torch.sum(mat_loss_y)) /
                     (shape[0] * shape[1] * shape[2] * shape[3]))
        # mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y))
        return mean_loss



class BayesianSGD(Optimizer):

    def __init__(self, params, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(BayesianSGD, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(BayesianSGD, self).__setstate__(state)

        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            # uncertainty = group['uncertaintsy']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if isinstance(group['lr'], torch.Tensor):
                    p.data = p.data + torch.mul(-group['lr'].data, d_p)
                else:
                    p.data.add_(-group['lr'], d_p)

        return loss

class Attention_gate(nn.Module):
    def __init__(self, in_planes_1, in_planes_2, out_planes):
        super().__init__()
        self.theta = nn.Conv3d(in_planes_1, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = nn.Conv3d(in_planes_2, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        # self.psi = nn.Conv3d(out_planes, 1, kernel_size=1, stride=1, padding=0, bias=True)
        # self.W_out = nn.Conv3d(in_planes_1, in_planes_1, kernel_size=1, stride=1, padding=0, bias=True)
        # self.norm = nn.InstanceNorm3d(in_planes_1, affine=True)

    def forward(self, x_l, x_g):
        input_size = x_l.size()

        theta_x = self.theta(x_l)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(x_g), size=theta_x_size[-3:], mode='trilinear', align_corners=True)

        # sum, relu
        f = F.relu(theta_x + phi_g, inplace=True)

        # out, sigmoid
        # sigm_psi_f = torch.sigmoid(self.psi(f))

        # interporlate, gate
        # sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[-3:], mode='trilinear', align_corners=False)
        # y = sigm_psi_f.expand_as(x_l) * x_l

        # out, norm, actf
        # y = self.norm(self.W_out(y))

        return f

class Attention_gate_2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.theta = nn.Conv3d(in_planes, in_planes//2, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv3d(in_planes//2, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.out = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(out_planes, affine=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        theta_x = self.theta(x)
        f = F.leaky_relu(theta_x, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))
        y = sigm_psi_f.expand_as(x) * x
        y = self.out(y)
        return y

class Attention_gate_3(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_planes // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gate_output = self.gate(x)
        return x * gate_output

class Attention_gate_4(nn.Module):
    def __init__(self, in_planes_1, in_planes_2, out_planes):
        super().__init__()
        self.theta = nn.Conv3d(in_planes_1, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.phi = nn.Conv3d(in_planes_2, out_planes, kernel_size=1, stride=1, padding=0, bias=True)

        self.psi = nn.Conv3d(out_planes, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x_l, x_g):
        """ embed """
        theta_x = self.theta(x_l)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(x_g), size=theta_x_size[-3:], mode='trilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        """ gate """
        sigm_psi_f = torch.sigmoid(self.psi(f))

        """ apply gate """
        # sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[-3:], mode='trilinear')
        # y = sigm_psi_f.expand_as(x_l) * x_l

        y = x_l * (1 - sigm_psi_f * self.gamma)

        return y


class Attention_gate_5(nn.Module):
    def __init__(self, n_featuremap_q, n_featuremap_kv, n_heads = 4, d_k = 16):
        super(Attention_gate_5, self).__init__()
        self.num_heads = n_heads
        self.d_k = d_k

        self.query_conv = nn.Conv1d(n_featuremap_q, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=True)
        self.key_conv = nn.Conv1d(n_featuremap_kv, self.num_heads * self.d_k, kernel_size=1, padding=0, bias=True)
        self.value_conv = nn.Conv1d(n_featuremap_kv, self.num_heads * self.d_k, kernel_size=1, bias=True)
        self.output_conv = nn.Conv1d(self.num_heads * self.d_k, n_featuremap_q, kernel_size=1, bias=True)

        """ gamma """
        self.gamma = nn.Parameter(torch.zeros(1))
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, x_q, x_kv):
        m_batchsize, C, width = x_q.size()
        total_q_depth = width

        m_batchsize, C, width_kv = x_kv.size()
        total_kv_depth = width_kv

        """ linear for each component"""
        queries = self.query_conv(x_q).view(m_batchsize, -1, total_q_depth).permute(0, 2, 1)
        keys = self.key_conv(x_kv).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)
        values = self.value_conv(x_kv).view(m_batchsize, -1, total_kv_depth).permute(0, 2, 1)    # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries) # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """Combine queries and keys """
        logits = torch.matmul(queries, keys.permute(0,1,3,2))
        weights = torch.sigmoid(logits)
        out = torch.matmul(weights, values)

        """ merge heads """
        out = self._merge_heads(out)

        """ linear to get output """
        out = out.view(m_batchsize, width, -1).permute(0, 2, 1)
        out = torch.sum(out, dim=-1, keepdim=True) / (total_q_depth * total_kv_depth)
        out = self.output_conv(out)

        """ residual """
        # out = self.gamma * out
        # out = out + x_q

        # x = (1-self.gamma) * x
        # out = torch.cat((out, x), 1)

        # out = self.act_f(out)
        return out


class DoubleAttentionLayer(nn.Module):
    """
    Implementation of Double Attention Network. NIPS 2018
    """

    def __init__(self, in_channels: int, c_m: int, c_n: int, reconstruct=False):
        """
        Parameters
        ----------
        in_channels
        c_m
        c_n
        reconstruct: `bool` whether to re-construct output to have shape (B, in_channels, L, R)
        """
        super(DoubleAttentionLayer, self).__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.convA = nn.Conv3d(in_channels, c_m, kernel_size=1)
        self.convB = nn.Conv3d(in_channels, c_n, kernel_size=1)
        self.convV = nn.Conv3d(in_channels, c_n, kernel_size=1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv3d(c_m, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: `torch.Tensor` of shape (B, C, H, W)
        Returns
        -------
        """
        batch_size, c, d, h, w = x.size()
        assert c == self.in_channels, 'input channel not equal!'
        A = self.convA(x)  # (B, c_m, d, h, w) because kernel size is 1
        B = self.convB(x)  # (B, c_n, d, h, w)
        V = self.convV(x)  # (B, c_n, d, h, w)
        tmpA = A.view(batch_size, self.c_m, d * h * w)
        attention_maps = B.view(batch_size, self.c_n, d * h * w)
        attention_vectors = V.view(batch_size, self.c_n, d * h * w)

        attention_maps = F.softmax(attention_maps, dim=-1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim=1)  # (B, c_n, h * w) attention on c_n dimension
        tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
        tmpZ = tmpZ.view(batch_size, self.c_m, d, h, w)
        if self.reconstruct: tmpZ = self.conv_reconstruct(tmpZ)

        tmpZ = F.relu(tmpZ, inplace=True)
        return tmpZ



class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln) # 256, 128, 256, 8
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class MAB_1(nn.Module):
    def __init__(self, dim_Q, dim_KV, dim_out, num_heads=4, ln=False):
        super(MAB_1, self).__init__()
        self.num_heads = num_heads
        self.dim_v = dim_out
        self.d_k = dim_out // num_heads

        """ query """
        self.query_conv = nn.Conv1d(dim_Q, dim_out, kernel_size=1, bias=True)
        self.key_conv = nn.Conv1d(dim_KV, dim_out, kernel_size=1, bias=True)
        self.value_conv = nn.Conv1d(dim_KV, dim_out, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.output_conv = nn.Conv1d(dim_out, dim_out, kernel_size=1, bias=True)
        self.act_f = nn.ReLU(inplace=True)
        if ln:
            # self.ln0 = nn.LayerNorm(dim_out, elementwise_affine=True)
            # self.ln1 = nn.LayerNorm(dim_out, elementwise_affine=True)
            self.ln0 = LayerNorm(dim_out)
            self.ln1 = LayerNorm(dim_out)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, Q, K):
        m_batchsize, _, spatial_dim_Q = Q.size()
        m_batchsize, _, spatial_dim_K = K.size()

        """ linear for each component"""
        queries = self.query_conv(Q).view(m_batchsize, -1, spatial_dim_Q).permute(0, 2, 1)
        keys = self.key_conv(K).view(m_batchsize, -1, spatial_dim_K).permute(0, 2, 1)
        values = self.value_conv(K).view(m_batchsize, -1, spatial_dim_K).permute(0, 2, 1)  # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries)  # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ attention gaussian embedding """
        query_scale = np.power(self.d_k, -0.5)
        logits = torch.matmul(queries * query_scale, keys.permute(0, 1, 3, 2))
        weights = self.softmax(logits)  # BX (N) X (N/p)

        """ dot product """
        # logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        # weights = logits / logits.size(-1)

        """ value matmul """
        out = queries + torch.matmul(weights, values)  # B, n_h, N, f
        # out = torch.matmul(weights, values)
        # out = torch.cat([out, queries], dim=-1)

        """ merge heads """
        out = self._merge_heads(out)
        out = out.view(m_batchsize, spatial_dim_Q, -1).permute(0, 2, 1)  # B, f, seed

        out = out if getattr(self, 'ln0', None) is None else self.ln0(out)
        # out = out if getattr(self, 'ln0', None) is None else self.ln0(out.transpose(1, 2)).transpose(1, 2)

        out = out + self.act_f(self.output_conv(out))
        out = out if getattr(self, 'ln1', None) is None else self.ln1(out)
        # out = out if getattr(self, 'ln1', None) is None else self.ln1(out.transpose(1, 2)).transpose(1, 2)

        return out


class ISAB_1(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB_1, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, dim_out, num_inds))
        nn.init.kaiming_normal_(self.I, nonlinearity='relu')

        self.mab0 = MAB_1(dim_Q=dim_out, dim_KV=dim_in, dim_out=dim_out, num_heads=num_heads, ln=ln)
        self.mab1 = MAB_1(dim_Q=dim_in, dim_KV=dim_out, dim_out=dim_out, num_heads=num_heads, ln=ln)

    def forward(self, X_q, X_k):
        H = self.mab0(self.I.repeat(X_k.size(0), 1, 1), X_k)
        return self.mab1(X_q, H)


class PMA_1(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA_1, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, dim, num_seeds))
        nn.init.kaiming_normal_(self.S, nonlinearity='relu')
        self.mab = MAB_1(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SAB_1(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB_1, self).__init__()
        self.mab = MAB_1(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class MAB_2(nn.Module):
    def __init__(self, dim_Q, dim_KV, d_k, num_heads=4, ln=False):
        super(MAB_2, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k

        """ query """
        self.query_conv = nn.Conv1d(dim_Q, d_k * num_heads, kernel_size=1, bias=True)
        self.key_conv = nn.Conv1d(dim_KV, d_k * num_heads, kernel_size=1, bias=True)
        self.value_conv = nn.Conv1d(dim_KV, d_k * num_heads, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Conv1d(d_k * num_heads, dim_Q, kernel_size=1, bias=True)
        self.pwff = nn.Sequential(
            nn.Conv1d(dim_Q, dim_Q * 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_Q * 2, dim_Q, kernel_size=1, bias=True),
        )
        # self.alpha = nn.Parameter(torch.ones(1))
        if ln:
            # self.ln0 = nn.LayerNorm(dim_Q, elementwise_affine=True)
            # self.ln1 = nn.LayerNorm(dim_Q, elementwise_affine=True)

            # self.ln0_q = LayerNorm(dim_Q)
            # self.ln0_k = LayerNorm(dim_Q)
            # self.ln1 = LayerNorm(dim_Q)

            self.ln0_q = nn.InstanceNorm1d(dim_Q, affine=True)
            self.ln0_k = nn.InstanceNorm1d(dim_Q, affine=True)
            self.ln1 = nn.InstanceNorm1d(dim_Q, affine=True)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return (x.permute(0, 2, 1, 3).contiguous()
                .view(shape[0], shape[2], shape[3] * self.num_heads))

    def forward(self, Q, K, mask):
        m_batchsize, _, spatial_dim_Q = Q.size()
        m_batchsize, _, spatial_dim_K = K.size()

        Q_norm = Q if getattr(self, 'ln0', None) is None else self.ln0_q(Q)
        # out = out if getattr(self, 'ln0', None) is None else self.ln0(out.transpose(1, 2)).transpose(1, 2)

        K_norm = K if getattr(self, 'ln0', None) is None else self.ln0_k(K)
        # out = out if getattr(self, 'ln0', None) is None else self.ln0(out.transpose(1, 2)).transpose(1, 2)

        """ linear for each component"""
        queries = self.query_conv(Q_norm).view(m_batchsize, -1, spatial_dim_Q).permute(0, 2, 1)
        keys = self.key_conv(K_norm).view(m_batchsize, -1, spatial_dim_K).permute(0, 2, 1)
        values = self.value_conv(K_norm).view(m_batchsize, -1, spatial_dim_K).permute(0, 2, 1)  # B X N X Channel

        """ split into multiple heads """
        queries = self._split_heads(queries)  # B, head, N, d_k
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        """ attention gaussian embedding """
        # logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        query_scale = np.power(self.d_k, -0.5)
        logits = torch.matmul(queries * query_scale, keys.permute(0, 1, 3, 2))

        if mask is not None:
            # pass
            # logits -= (torch.log(1.0 - mask.unsqueeze(-2) + 1e-7))
            logits -= ((1.0 - mask.unsqueeze(-2)) * 10000)
            # logits -= (torch.log(1.0 - mask.unsqueeze(-2) + 1e-7) * 1000)
        # print(self.alpha)

        weights = torch.sigmoid(logits)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        # weights = self.softmax(logits)  # BX (N) X (N/p)

        """ dot product """
        # logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        # weights = logits / logits.size(-1)
        # weights = torch.tanh(logits)
        # weights = weights / weights.sum(dim=-1, keepdims=True)

        """ value matmul """
        # out = queries + torch.matmul(weights, values)  # B, n_h, N, f
        out = torch.matmul(weights, values)
        # out = torch.cat([out, queries], dim=-1)

        """ merge heads """
        out = self._merge_heads(out)
        out = out.view(m_batchsize, spatial_dim_Q, -1).permute(0, 2, 1)  # B, f, seed

        # out = self.proj(out)
        out = self.proj(out) + Q

        """ normalize """
        out_res = out
        out = out if getattr(self, 'ln1', None) is None else self.ln1(out)
        # out = out if getattr(self, 'ln1', None) is None else self.ln1(out.transpose(1, 2)).transpose(1, 2)
        out = self.pwff(out) + out_res


        return out


class ISAB_2(nn.Module):
    def __init__(self, dim_in, dim_seed, d_k, num_heads, num_inds, ln=False):
        super(ISAB_2, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, dim_seed, num_inds))
        nn.init.kaiming_normal_(self.I, nonlinearity='relu')
        # nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB_2(dim_Q=dim_seed, dim_KV=dim_in, d_k=d_k, num_heads=num_heads, ln=ln)
        self.mab1 = MAB_2(dim_Q=dim_in, dim_KV=dim_seed, d_k=d_k, num_heads=num_heads, ln=ln)

    def forward(self, X_q, X_k, mask):
        H = self.mab0(self.I.repeat(X_k.size(0), 1, 1), X_k, mask)
        return self.mab1(X_q, H, None)


class PMA_2(nn.Module):
    def __init__(self, dim_in, dim_seed, d_k, num_heads, num_seeds, ln=False):
        super(PMA_2, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, dim_in, num_seeds))
        nn.init.kaiming_normal_(self.S, nonlinearity='relu')
        # nn.init.xavier_uniform_(self.S)
        self.mab = MAB_2(dim_seed, dim_in, d_k, num_heads, ln=ln)

    def forward(self, X, mask):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask)

class SAB_2(nn.Module):
    def __init__(self, dim_in, d_k, num_heads, ln=False):
        super(SAB_2, self).__init__()
        self.mab = MAB_2(dim_in, dim_in, d_k, num_heads, ln=ln)

    def forward(self, X, mask):
        return self.mab(X, X, mask)

class AgeEncoding(nn.Module):
    "Implement the AE function."

    def __init__(self, d_model, dropout, out_dim, max_len=240):
        super(AgeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  ## (240, 512)
        position = torch.arange(0, max_len).unsqueeze(1).float()  ## (240, 1)
        div_term = torch.exp((torch.arange(0, d_model, 2).float() *
                              -(math.log(10000.0) / d_model)).float())  ## (256)
        pe[:, 0::2] = torch.sin(position * div_term)  ## (240, 256)
        pe[:, 1::2] = torch.cos(position * div_term)  ## (240, 256)
        self.register_buffer('pe', pe)
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1', nn.Linear(d_model, 64))
        self.fc6.add_module('lrn0_s1', nn.LayerNorm(64))
        self.fc6.add_module('fc6_s3', nn.Linear(64, out_dim))

    def forward(self, x, age_id):
        age_id = torch.round(age_id * 2)
        age_id = age_id.long()
        y = []
        for i_batch in range(x.size(0)):
            y.append(torch.autograd.Variable(self.pe[age_id[i_batch], :], requires_grad=False).unsqueeze(0))
        y = torch.cat(y, dim=0)
        y = self.fc6(y)

        x = x + y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return self.dropout(x)



""" selection """
# list_batch = []
# for i_batch in range(x_0[0].size(0)):
#     for i_scale in range(len(action)):
#         index = (action[i_scale][i_batch].view(-1) == 1).nonzero().squeeze(1)
#         tmp_x = torch.index_select(x_0[i_scale][i_batch].view(x_0[i_scale][i_batch].size(0), -1), -1, index).unsqueeze(
#             0)
#
#         tmp_x = torch.mean(tmp_x, dim=-1, keepdim=True)
#         list_batch.append(tmp_x)

