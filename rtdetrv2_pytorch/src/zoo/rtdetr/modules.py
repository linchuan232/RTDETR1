import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


__all__ = ['CAFMFusion','DenoiseBasicBlock','LightweightDenoiseBlock']



import torch
import torch.nn as nn
import numbers
from einops import rearrange
from collections import OrderedDict


# ==================== 辅助函数 ====================
def to_3d(x):
    """将4维张量转换为3维张量（空间维度展平）
    输入: [B, C, H, W] -> 输出: [B, H*W, C]
    """
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """将3维张量恢复为4维张量（还原空间维度）
    输入: [B, H*W, C] -> 输出: [B, C, H, W]
    """
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def resize_complex_weight(origin_weight, new_h, new_w):
    """调整复数权重的尺寸（用于动态滤波器适配不同输入尺寸）
    
    Args:
        origin_weight: 原始权重 [h, w, num_heads, 2]
        new_h: 目标高度
        new_w: 目标宽度
    
    Returns:
        调整后的权重 [new_h, new_w, num_heads, 2]
    """
    h, w, num_heads = origin_weight.shape[0:3]
    # 调整维度为 [1, num_heads*2, h, w]
    origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
    # 双三次插值调整尺寸
    new_weight = torch.nn.functional.interpolate(
        origin_weight,
        size=(new_h, new_w),
        mode='bicubic',
        align_corners=True
    )
    # 恢复原始维度顺序
    new_weight = new_weight.permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
    return new_weight


# ==================== 归一化层 ====================
class BiasFree_LayerNorm(nn.Module):
    """无偏置的层归一化（LayerNorm）实现"""
    
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        assert len(self.normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """带偏置的层归一化实现（标准LayerNorm）"""
    
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        assert len(self.normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """层归一化封装类（可选择是否带偏置）"""
    
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ==================== 激活函数 ====================
class StarReLU(nn.Module):
    """自定义激活函数 StarReLU: s * relu(x)² + b
    
    相比传统ReLU，StarReLU提供了更强的非线性能力，
    同时保持了计算效率和训练稳定性。
    """
    
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * (self.relu(x) ** 2) + self.bias


# ==================== MLP 模块 ====================
class Mlp(nn.Module):
    """多层感知机（MLP）
    
    标准的两层全连接网络，常用于Transformer和现代CNN架构中。
    """
    
    def __init__(self, dim, mlp_ratio=4, out_features=None, 
                 act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ==================== CID 模块（通道独立-依赖模块）====================
class DConv7(nn.Module):
    """7x7 深度可分离卷积
    
    使用较大的卷积核（7x7）提供更大的感受野，
    通过深度卷积（groups=channels）保持参数效率。
    """
    
    def __init__(self, f_number, padding_mode='reflect'):
        super().__init__()
        self.dconv = nn.Conv2d(
            f_number, f_number, 
            kernel_size=7, 
            padding=3,
            groups=f_number,  # 深度卷积：每个通道独立卷积
            padding_mode=padding_mode
        )
    
    def forward(self, x):
        return self.dconv(x)


class ChannelMLP(nn.Module):
    """通道MLP（用于CID模块）
    
    通过1x1卷积实现的MLP，用于建模通道间的依赖关系。
    """
    
    def __init__(self, f_number, excitation_factor=2):
        super().__init__()
        self.act = nn.GELU()
        self.pwconv1 = nn.Conv2d(f_number, excitation_factor * f_number, kernel_size=1)
        self.pwconv2 = nn.Conv2d(f_number * excitation_factor, f_number, kernel_size=1)
    
    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x


class CID(nn.Module):
    """通道独立-依赖（Channel Independent-Dependent）模块
    
    结合了：
    1. 通道独立操作（深度卷积）：捕获空间特征
    2. 通道依赖操作（MLP）：建模通道间关系
    """
    
    def __init__(self, f_number):
        super().__init__()
        self.channel_independent = DConv7(f_number)
        self.channel_dependent = ChannelMLP(f_number)
    
    def forward(self, x):
        return self.channel_dependent(self.channel_independent(x))


# ==================== 动态滤波器模块 ====================
class DynamicFilter(nn.Module):
    """动态滤波器模块（频域特征操作）
    
    核心思想：
    1. 将空域特征转换到频域（FFT）
    2. 使用可学习的复数滤波器在频域进行滤波
    3. 通过动态路由机制选择最优滤波器组合
    4. 转换回空域（IFFT）
    
    优势：
    - 有效抑制频域噪声
    - 自适应调整滤波策略
    - 计算高效（FFT是O(nlogn)）
    """
    
    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=30, weight_resize=True,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1  # FFT后的频域尺寸
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        
        # 特征扩展
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        
        # 动态路由网络：生成滤波器权重
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        
        # 可学习的复数滤波器（频域权重）
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02
        )
        
        # 特征压缩
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x, y):
        """
        Args:
            x: 输入特征 [B, H, W, C]
            y: 条件特征 [B, H, W, C]（用于生成动态路由权重）
        
        Returns:
            滤波后的特征 [B, H, W, C]
        """
        B, H, W, _ = x.shape
        
        # 1. 生成动态路由权重
        routeing = self.reweight(y.mean(dim=(1, 2)))  # 全局平均池化
        routeing = routeing.view(B, self.num_filters, -1).softmax(dim=1)
        
        # 2. 特征扩展
        x = self.pwconv1(x)
        x = self.act1(x)
        
        # 3. 转换到频域（实值FFT）
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        # 4. 准备复数滤波器
        if self.weight_resize:
            complex_weights = resize_complex_weight(
                self.complex_weights, x.shape[1], x.shape[2]
            )
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        
        # 5. 动态加权滤波器
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        
        # 6. 频域滤波（复数乘法）
        x = x * weight
        
        # 7. 转换回空域（逆FFT）
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        
        # 8. 特征压缩
        x = self.act2(x)
        x = self.pwconv2(x)
        
        return x


# ==================== 噪声感知注意力模块 ====================
class NoiseAwareAttention(nn.Module):
    """噪声感知注意力机制
    
    通过同时使用平均池化和最大池化来捕获噪声模式：
    - 平均池化：捕获整体噪声水平
    - 最大池化：捕获局部噪声峰值
    
    生成注意力权重来自适应抑制噪声区域。
    """
    
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(dim * 2, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            加权后的特征 [B, C, H, W]
        """
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.fc(combined)
        return x * attention


# ==================== 基础卷积层 ====================
class ConvNormLayer(nn.Module):
    """基础卷积+归一化层
    
    标准的 Conv -> BN -> Act 组合。
    """
    
    def __init__(self, ch_in, ch_out, kernel_size, stride, act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride,
            padding=(kernel_size - 1) // 2, 
            bias=False
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = self._get_activation(act) if act else nn.Identity()
    
    def _get_activation(self, act):
        if act == 'relu':
            return nn.ReLU(inplace=True)
        elif act == 'gelu':
            return nn.GELU()
        elif act == 'starrelu':
            return StarReLU()
        else:
            return nn.Identity()
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ==================== 去噪增强的 BasicBlock ====================
class DenoiseBasicBlock(nn.Module):
    """去噪增强的 BasicBlock（完整版）
    
    架构设计：
    ┌─────────────────────────────────────────────────┐
    │  输入特征                                        │
    └────────┬────────────────────────────────────────┘
             │
             ├──→ 3x3 Conv (stride, 通道变换)
             │
             ├──→ CID (空域特征增强)
             │    ├─ 7x7 深度卷积
             │    └─ 通道MLP
             │
             ├──→ DynamicFilter (频域去噪) [可选]
             │    ├─ FFT 转频域
             │    ├─ 动态滤波器加权
             │    └─ IFFT 转空域
             │
             ├──→ 特征融合 (空域+频域)
             │
             ├──→ NoiseAwareAttention (注意力增强) [可选]
             │
             └──→ 残差连接 + 激活
                  │
                  ▼
             输出特征
    
    特性：
    1. 频域动态滤波 - 自适应抑制频域噪声
    2. CID 空域增强 - 增强纹理和结构信息
    3. 噪声感知注意力 - 自适应加权干净/噪声区域
    4. 灵活配置 - 可根据需求开关各模块
    
    使用场景：
    - 低光照图像处理
    - 医学图像去噪
    - 视频去噪
    - 任何需要鲁棒特征提取的任务
    """
    
    expansion = 1
    
    def __init__(self, ch_in, ch_out, stride, shortcut, 
                 act='gelu', variant='b', 
                 use_frequency_filter=True,
                 use_noise_attention=True,
                 num_filters=4,
                 filter_size=30):
        """
        Args:
            ch_in: 输入通道数
            ch_out: 输出通道数
            stride: 步长（1或2）
            shortcut: 是否使用恒等shortcut
            act: 激活函数类型 ('relu', 'gelu', 'starrelu')
            variant: ResNet变体 ('b' or 'd')
            use_frequency_filter: 是否使用频域滤波器
            use_noise_attention: 是否使用噪声感知注意力
            num_filters: 动态滤波器数量
            filter_size: 滤波器基础尺寸
        """
        super().__init__()
        self.shortcut = shortcut
        self.use_frequency_filter = use_frequency_filter
        self.use_noise_attention = use_noise_attention
        
        # ========== Shortcut 连接 ==========
        if not shortcut:
            if variant == 'd' and stride == 2:
                # ResNet-D 变体：使用平均池化
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1, act=None))
                ]))
            else:
                # 标准shortcut：1x1卷积
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride, act=None)
        
        # ========== 主分支：空域特征提取 ==========
        # 第一层：3x3卷积（处理stride和初步特征提取）
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        
        # 第二层：CID模块（空间特征增强）
        self.cid = CID(ch_out)
        self.cid_norm = nn.BatchNorm2d(ch_out)
        
        # ========== 频域滤波分支（可选）==========
        if use_frequency_filter:
            # 自适应调整滤波器尺寸
            adaptive_size = min(filter_size, ch_out)
            
            self.freq_filter = DynamicFilter(
                dim=ch_out,
                expansion_ratio=1,
                reweight_expansion_ratio=0.25,
                num_filters=num_filters,
                size=adaptive_size,
                weight_resize=True,
                act1_layer=StarReLU,
                act2_layer=nn.Identity
            )
            self.freq_norm = LayerNorm(ch_out, 'WithBias')
        
        # ========== 噪声感知注意力（可选）==========
        if use_noise_attention:
            self.noise_attention = NoiseAwareAttention(ch_out, reduction=8)
        
        # ========== 特征融合 ==========
        fusion_channels = ch_out * (2 if use_frequency_filter else 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, ch_out, 1, bias=False),
            nn.BatchNorm2d(ch_out)
        )
        
        # ========== 最终激活函数 ==========
        self.act = self._get_activation(act)
    
    def _get_activation(self, act):
        """获取激活函数"""
        if act == 'relu':
            return nn.ReLU(inplace=True)
        elif act == 'gelu':
            return nn.GELU()
        elif act == 'starrelu':
            return StarReLU()
        else:
            return nn.Identity()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            输出特征 [B, C', H', W']
        """
        # ========== 阶段1：空域卷积 + CID ==========
        out = self.branch2a(x)
        spatial_feat = self.cid(out)
        spatial_feat = self.cid_norm(spatial_feat)
        
        # ========== 阶段2：频域滤波（可选）==========
        if self.use_frequency_filter:
            B, C, H, W = spatial_feat.shape
            
            # 转换为 [B, H, W, C] 格式（DynamicFilter要求）
            freq_input = spatial_feat.permute(0, 2, 3, 1)
            
            # 频域滤波（使用自身作为条件）
            freq_feat = self.freq_filter(freq_input, freq_input)
            
            # 转回 [B, C, H, W]
            freq_feat = freq_feat.permute(0, 3, 1, 2)
            freq_feat = self.freq_norm(freq_feat)
            
            # 融合空域和频域特征
            combined = torch.cat([spatial_feat, freq_feat], dim=1)
            out = self.fusion(combined)
        else:
            out = spatial_feat
        
        # ========== 阶段3：噪声感知注意力（可选）==========
        if self.use_noise_attention:
            out = self.noise_attention(out)
        
        # ========== 阶段4：残差连接 ==========
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)
        
        return out


# ==================== 轻量级去噪模块 ====================
class LightweightDenoiseBlock(nn.Module):
    """轻量级去噪 BasicBlock
    
    只使用 CID + 噪声注意力，不使用频域滤波。
    适合资源受限场景（移动端、边缘设备）。
    
    参数量约为完整版的 30-40%。
    """
    
    expansion = 1
    
    def __init__(self, ch_in, ch_out, stride, shortcut, 
                 act='gelu', variant='b'):
        super().__init__()
        self.shortcut = shortcut
        
        # Shortcut 连接
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1, act=None))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride, act=None)
        
        # 主分支
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.cid = CID(ch_out)
        self.cid_norm = nn.BatchNorm2d(ch_out)
        self.noise_attention = NoiseAwareAttention(ch_out, reduction=8)
        
        self.act = nn.GELU() if act == 'gelu' else nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.branch2a(x)
        out = self.cid(out)
        out = self.cid_norm(out)
        out = self.noise_attention(out)
        
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)
        return out

'''
二次创新模块：CGAFusion（2024 TIP顶刊） 结合 CAFM（2024 SCI 二区）：CAFMFusion用于高频与低频特征校准/融合模块 （冲二，三，保四）

CAFM:所提出的卷积和注意力特征融合模块。它由局部和全局分支组成。
    在局部分支中，采用卷积和通道洗牌进行局部特征提取。
    在全局分支中，注意力机制用于对长程特征依赖关系进行建模。

CGAFusion（2024 TIP顶刊）:我们提出了一种新的注意机制，可以强调用特征编码的更多有用的信息，以有效地提高性能。
    此外，还提出了一种基于CGA的混合融合方案，可以有效地将编码器部分的低级特征与相应的高级特征进行融合。

强强联手：CGAFusion（2024 TIP顶刊） 结合 CAFM（2024 SCI 二区）：CAFMFusion
        CAFMFusion用于低级特征与高级特征校准/融合模块 （冲二，三，保四）
适用于：图像去噪，图像增强，目标检测，语义分割，实例分割，图像恢复，暗光增强等所有CV2维任务
'''
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
class CAFM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True, groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)
        out_conv = out_conv.squeeze(2)

        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv
        return output

class CAFMFusion(nn.Module):
    def __init__(self, dim):
        super(CAFMFusion, self).__init__()
        self.CAFM = CAFM(dim)
        self.PixelAttention = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f):
        x,y = f
        initial = x + y
        pattn1 = self.CAFM(initial)
        pattn2 = self.sigmoid(self.PixelAttention(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

