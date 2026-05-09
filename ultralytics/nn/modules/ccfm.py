import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # 为了防止缩放后通道数太少导致除以 reduction 报错，可以加一个 max 保护
        mid_channel = max(channel // reduction, 8) 
        self.fc1 = nn.Conv2d(channel, mid_channel, 1)
        self.fc2 = nn.Conv2d(mid_channel, channel, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class CCFM(nn.Module):
    # c1 是 Ultralytics 自动传入的输入通道列表 (例如 [512, 1024])
    # c2 是 Ultralytics 根据模型 width 自动缩放后的输出通道数
    def __init__(self, c1, c2):
        super().__init__()
        
        # 不再硬编码 256，而是使用动态缩放后的 c2 作为隐藏层和输出层的通道
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(c, c2, 1) for c in c1
        ])
        
        self.se_blocks = nn.ModuleList([
            SEBlock(c2) for _ in c1
        ])
        
        # 拼接后的通道数是 c2 乘以输入的个数，最终通过 1x1 卷积输出 c2
        self.out_conv = nn.Conv2d(c2 * len(c1), c2, 1)

    def forward(self, features):
        resized_feats = []
        target_size = features[0].shape[2:]

        for i, feat in enumerate(features):
            x = self.scale_convs[i](feat)
            x = F.interpolate(x, size=target_size, mode='nearest') if x.shape[2:] != target_size else x
            x = self.se_blocks[i](x)
            resized_feats.append(x)

        fused = torch.cat(resized_feats, dim=1)
        out = self.out_conv(fused)
        return out