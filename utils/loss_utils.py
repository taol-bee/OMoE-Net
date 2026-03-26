import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from einops import rearrange


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                     tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or(self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


    
    
class OrthogonalLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(OrthogonalLoss, self).__init__()
        self.eps = eps

    def forward(self, expert_features):
        num_experts = len(expert_features)
        if num_experts <= 1:
            return torch.tensor(0.0, device=expert_features[0].device)
        
        total_loss = 0.0
        count = 0
        
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                # [B, C, H, W] -> [B, C, L]
                feat_i = expert_features[i].flatten(2)
                feat_j = expert_features[j].flatten(2)
                
                # 归一化
                norm_i = torch.norm(feat_i, dim=2, keepdim=True) + self.eps
                norm_j = torch.norm(feat_j, dim=2, keepdim=True) + self.eps
                normalized_i = feat_i / norm_i
                normalized_j = feat_j / norm_j
                
                # 计算通道相似度 [B, C]
                channel_similarity = torch.sum(normalized_i * normalized_j, dim=2)
                
                # === 关键修改 ===
                # 原来: mean -> square (导致正负抵消，值极小)
                # 现在: square -> mean (强制每个通道正交，值恢复正常量级)
                total_loss += torch.mean(torch.square(channel_similarity))
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0, device=expert_features[0].device)
    
    
class FrequencyLoss(nn.Module):
    def __init__(self, loss_weight=0.1):
        super(FrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # 转换到频域
        pred_fft = torch.fft.rfft2(pred, norm='backward')
        target_fft = torch.fft.rfft2(target, norm='backward')
        
        # 计算频域的 L1 距离 (实部+虚部)
        loss = self.criterion(torch.view_as_real(pred_fft), torch.view_as_real(target_fft))
        return self.loss_weight * loss
    
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss