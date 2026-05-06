# Adapt from MIOFlow

__all__ = ['MMD_loss', 'OT_loss', 'Density_loss', 'Local_density_loss']

import os, math, numpy as np
import torch
import torch.nn as nn
# Adapt from MIOFlow
class MMD_loss(nn.Module):
    '''
    https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py
    '''
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size].mean()
        YY = kernels[batch_size:, batch_size:].mean()
        XY = kernels[:batch_size, batch_size:].mean()
        YX = kernels[batch_size:, :batch_size].mean()
        loss = XX + YY - XY -YX
        return loss

# Adapt from MIOFlow with modification
import ot
import torch.nn as nn
import torch
import numpy as np

class OT_loss1(nn.Module):
    _valid = 'emd sinkhorn sinkhorn_knopp_unbalanced'.split()

    def __init__(self, which='emd', device=None):
        if which not in self._valid:
            raise ValueError(f'{which} not known ({self._valid})')
        self.which = which
        self.device = device

    def __call__(self, source, target, mu, nu, sigma=None):
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32)
        if not isinstance(nu, torch.Tensor):
            nu = torch.tensor(nu, dtype=torch.float32)
        
        if self.device:
            #mu = mu.cuda()
            mu = mu.to(self.device)
            nu = nu.to(self.device)

        M = torch.cdist(source, target)**2

        if self.which == 'emd':
            pi = ot.emd(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy())
        elif self.which == 'sinkhorn':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn method')
            #pi = ot.sinkhorn(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy(), sigma.detach().cpu().numpy(), method='sinkhorn_log')
            #pi = ot.sinkhorn(mu, nu, M, sigma, method='sinkhorn_log')
            pi = ot.sinkhorn(mu, nu, M, sigma)
        elif self.which == 'sinkhorn_knopp_unbalanced':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn_knopp_unbalanced method')
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy(), sigma, sigma)
        else:
            raise ValueError(f'{self.which} not known ({self._valid})')

        if isinstance(pi, np.ndarray):
            pi = torch.tensor(pi, dtype=torch.float32)
        elif isinstance(pi, torch.Tensor):
            pi = pi.clone().detach()
        
        #pi = pi.cuda() if use_cuda else pi
        pi = pi.to(self.device)
        M = M.to(pi.device)
        loss = torch.sum(pi * M)
        return loss

import torch
import torch.nn as nn
import ot

class OT_loss2(nn.Module):
    _valid = 'emd sinkhorn sinkhorn_knopp_unbalanced'.split()

    def __init__(self, which='emd', alpha_spatial=1, alpha_express=1, use_cuda=True):
        """
        初始化 OT_loss2 类。
        
        参数:
            which (str): OT 方法类型，可选 'emd', 'sinkhorn', 或 'sinkhorn_knopp_unbalanced'。
            alpha (float): 控制第一组特征（前两列）的权重，范围 [0, 1]，默认 0.5。
            use_cuda (bool): 是否使用 GPU（这里使用 'mps' 设备），默认 True。
        """
        if which not in self._valid:
            raise ValueError(f'{which} not known ({self._valid})')
        self.which = which
        self.alpha_spatial = alpha_spatial  # 加权参数
        self.alpha_express=alpha_express
        self.use_cuda = use_cuda

    def __call__(self, source, target, mu, nu, sigma=None, use_cuda=None):
        """
        计算 OT loss。
        
        参数:
            source (torch.Tensor): 源数据，形状 (n, d)。
            target (torch.Tensor): 目标数据，形状 (m, d)。
            mu (torch.Tensor): 源分布的权重。
            nu (torch.Tensor): 目标分布的权重。
            sigma (float, optional): Sinkhorn 方法的正则化参数。
            use_cuda (bool, optional): 是否使用 GPU，默认为 None 时使用 self.use_cuda。
            
        返回:
            torch.Tensor: OT loss 值。
        """
        if use_cuda is None:
            use_cuda = self.use_cuda
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32)
        if not isinstance(nu, torch.Tensor):
            nu = torch.tensor(nu, dtype=torch.float32)
        
        if use_cuda:
            mu = mu.to('cuda')
            nu = nu.to('cuda')

        # 分割 source 和 target 为两组特征
        source_group1 = source[:, :2]  # 前两列特征
        source_group2 = source[:, 2:]  # 其余列特征
        target_group1 = target[:, :2]
        target_group2 = target[:, 2:]

        # # Calculate cost matrices for Gromov-Wasserstein between source and target
        # C1 = ot.dist(source_group1, source_group1)  # Cost matrix for source spatial features
        # C2 = ot.dist(target_group1, target_group1)  # Cost matrix for target spatial features
        
        # # Calculate Gromov-Wasserstein loss between source and target
        # gw_loss = ot.gromov.gromov_wasserstein2(
        #     C1, C2,  # Source and target cost matrices
        #     mu.detach(), nu.detach(),  # Source and target weights
        #     loss_fun='square_loss',
        #     log=False
        # )

        # 计算两组特征的距离矩阵
        M1 = ot.dist(source_group1, target_group1) # 第一组特征的距离矩阵
        M2 = ot.dist(source_group2, target_group2) # 第二组特征的距离矩阵
        # 加权组合成新的距离矩阵 M
        M = self.alpha_spatial * M1 + self.alpha_express * M2
        # M =M2
        # 根据指定的 OT 方法计算传输计划 pi
        if self.which == 'emd':
            #pi = ot.emd2(mu, nu, M)
            pi = ot.emd(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy())
        elif self.which == 'sinkhorn':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn method')
            pi = ot.sinkhorn(mu, nu, M, sigma)
        elif self.which == 'sinkhorn_knopp_unbalanced':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn_knopp_unbalanced method')
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), 
                                                         M.detach().cpu().numpy(), sigma, sigma)
        else:
            raise ValueError(f'{self.which} not known ({self._valid})')

        # 确保 pi 是张量并移动到正确设备
        if isinstance(pi, np.ndarray):
            pi = torch.tensor(pi, dtype=torch.float32)
        elif isinstance(pi, torch.Tensor):
            pi = pi.clone().detach()
        
        pi = pi.to('cuda') if use_cuda else pi
        M = M.to(pi.device)

        # 计算 OT loss
        loss = torch.sum(pi * M)
        #loss = loss * self.alpha_express + gw_loss * self.alpha_spatial
        return loss

class OT_loss3(nn.Module):
    _valid = 'emd sinkhorn sinkhorn_knopp_unbalanced'.split()

    def __init__(self, which='emd', alpha_spatial=1, alpha_express=1, use_cuda=True):
        """
        初始化 OT_loss2 类。
        
        参数:
            which (str): OT 方法类型，可选 'emd', 'sinkhorn', 或 'sinkhorn_knopp_unbalanced'。
            alpha (float): 控制第一组特征（前两列）的权重，范围 [0, 1]，默认 0.5。
            use_cuda (bool): 是否使用 GPU（这里使用 'mps' 设备），默认 True。
        """
        if which not in self._valid:
            raise ValueError(f'{which} not known ({self._valid})')
        self.which = which
        self.alpha_spatial = alpha_spatial  # 加权参数
        self.alpha_express=alpha_express
        self.use_cuda = use_cuda

    def __call__(self, source, target, mu, nu, sigma=None, use_cuda=None):
        """
        计算 OT loss。
        
        参数:
            source (torch.Tensor): 源数据，形状 (n, d)。
            target (torch.Tensor): 目标数据，形状 (m, d)。
            mu (torch.Tensor): 源分布的权重。
            nu (torch.Tensor): 目标分布的权重。
            sigma (float, optional): Sinkhorn 方法的正则化参数。
            use_cuda (bool, optional): 是否使用 GPU，默认为 None 时使用 self.use_cuda。
            
        返回:
            torch.Tensor: OT loss 值。
        """
        if use_cuda is None:
            use_cuda = self.use_cuda
        if not isinstance(mu, torch.Tensor):
            mu = torch.tensor(mu, dtype=torch.float32)
        if not isinstance(nu, torch.Tensor):
            nu = torch.tensor(nu, dtype=torch.float32)
        
        if use_cuda:
            mu = mu.to('cuda')
            nu = nu.to('cuda')

        # 分割 source 和 target 为两组特征
        source_group1 = source[:, :2]  # 前两列特征
        source_group2 = source[:, 2:]  # 其余列特征
        target_group1 = target[:, :2]
        target_group2 = target[:, 2:]

        # 计算两组特征的距离矩阵
        M1 = ot.dist(source_group1, target_group1) # 第一组特征的距离矩阵
        M2 = ot.dist(source_group2, target_group2) # 第二组特征的距离矩阵
        # 加权组合成新的距离矩阵 M
        M = self.alpha_spatial * M1 + self.alpha_express * M2
        # 根据指定的 OT 方法计算传输计划 pi
        if self.which == 'emd':
            pi = ot.emd2(mu, nu, M)
            #pi = ot.emd(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), M.detach().cpu().numpy())
        elif self.which == 'sinkhorn':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn method')
            pi = ot.sinkhorn(mu, nu, M, sigma)
        elif self.which == 'sinkhorn_knopp_unbalanced':
            if sigma is None:
                raise ValueError('sigma must be provided for sinkhorn_knopp_unbalanced method')
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(mu.detach().cpu().numpy(), nu.detach().cpu().numpy(), 
                                                         M.detach().cpu().numpy(), sigma, sigma)
        else:
            raise ValueError(f'{self.which} not known ({self._valid})')

        # 确保 pi 是张量并移动到正确设备
        if isinstance(pi, np.ndarray):
            pi = torch.tensor(pi, dtype=torch.float32)
        elif isinstance(pi, torch.Tensor):
            pi = pi
        
        pi = pi.to('cuda') if use_cuda else pi
        M = M.to(pi.device)

        # 计算 OT loss
        loss = pi
        return loss

import torch.nn as nn
import torch
class Density_loss(nn.Module):
    def __init__(self, hinge_value=0.01):
        self.hinge_value = hinge_value
        pass

    def __call__(self, source, target, groups = None, to_ignore = None, top_k = 5):
        if groups is not None:
            # for global loss
            c_dist = torch.stack([
                torch.cdist(source[i], target[i]) 
                # NOTE: check if this should be 1 indexed
                for i in range(1,len(groups))
                if groups[i] != to_ignore
            ])
        else:
            # for local loss
             c_dist = torch.stack([
                torch.cdist(source, target)                 
            ])
        values, _ = torch.topk(c_dist, top_k, dim=2, largest=False, sorted=False)
        values -= self.hinge_value
        values[values<0] = 0
        loss = torch.mean(values)
        return loss


class Local_density_loss(nn.Module):
    def __init__(self):
        pass

    def __call__(self, sources, targets, groups, to_ignore, top_k = 5):
        # print(source, target)
        # c_dist = torch.cdist(source, target) 
        c_dist = torch.stack([
            torch.cdist(sources[i], targets[i]) 
            # NOTE: check if should be from range 1 or not.
            for i in range(1, len(groups))
            if groups[i] != to_ignore
        ])
        vals, inds = torch.topk(c_dist, top_k, dim=2, largest=False, sorted=False)
        values = vals[inds[inds]]
        loss = torch.mean(values)
        return loss