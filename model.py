import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    2D Image Patch Embedding
    """
    def __init__(self, patch_size = 4, in_channels = 3, embedding_dim = 96, norm_layer = None):
        super(self, PatchEmbedding).__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.projection = nn.Conv2d(self.in_channels, self.embedding_dim, kernel_size = self.patch_size, stride = self.patch_size)
        self.norm = norm_layer(self.embedding_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        Perform patch embedding forward
        """
        _, _, h, w = x.shape

        # NOTE: 如果输入图片的宽高不是patch_size的整数倍, 需要padding操作。
        pad_input = (h % self.patch_size[0] != 0) or (w % self.patch_size[1] != 0)
        if pad_input:
            # 在图片的宽高维度进行padding
            # NOTE: pad是从最后一个维度依次向前padding, (w_left, w_right, h_top, h_bottom, c_front, c_back)
            x = F.pad(x, pad=(0, self.patch_size[1] - w % self.patch_size[1], 0, self.patch_size[0] - h % self.patch_size[0]))
        
        x = self.projection(x)
        _, _ , h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class PatchMerging(nn.Module):
    """
    2D Image Patch Merging Layer
    """
    def __init__(self, dim, norm_layer = nn.LayerNorm):
        super(PatchEmbedding, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias = False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, h, w):
        b, l, c = x.shape
        assert l == h * w, "input feature has wrong size"

        x = x.view(b, h, w, c)

        # NOTE: 如果输入的feature_map的宽高不是2的整数倍, 则需要padding。
        pad_input = (h % 2 == 1) or (w % 2 == 1)
        if pad_input:
            # NOTE: x: [b, h, w, c], 我们需要从宽高两个维度上进行padding。
            # (c_front, c_back, w_left, w_right, h_top, h_bottom)
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))

        # padding完之后, 需要进行merging操作
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim = -1)
        x = x.view(b, -1, 4 * c)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinTransformerBlock(nn.Module):
    """
    swin transformer block
    """
    def __init__(self, dim, num_heads, windows_size, shift_size, mlp_ratio, qkv_bias, drop_rate, norm_layer):
        pass


class BasicLayer(nn.Module):
    """
    a base swin transformer layer for each stage
    """
    def __init__(self, dim, depth, num_heads, windows_size, mlp_ratio, qkv_bias, drop_rate, norm_layer, downsample):
        super().__init__()
        self.dim = dim
        self.windows_size = windows_size
        # NOTE: 
        self.shift_size = windows_size // 2

        # build transformer block in current stage
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim = dim,
                num_heads = num_heads,
                windows_size = windows_size,
                shift_size = 0 if (i % 2 ==0) else self.shift_size,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                drop_rate = drop_rate,
                norm_layer = norm_layer
            )
         for i in range(depth)   
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim = dim, norm_layer = norm_layer)
        else:
            self.downsample = None


class SwinTransformerTiny(nn.Module):
    """
    Swin Transformer Tiny Version
    """
    def __init__(self, patch_size = 4, in_channels = 3, num_classes = 1000,
                embedding_dim = 96, depths = (2, 2, 6, 2), num_heads = (3, 6, 12, 24),
                window_size = 7, mlp_ratio = 4, qkv_bias = True, drop_rate = 0.1, 
                norm_layer = nn.LayerNorm, patch_norm = True):
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embedding_dim = embedding_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        # NOTE: 每经过一个stage, num_features就会变大两倍。
        self.num_features = int(embedding_dim * 2 ** (self.num_layers - 1))
        
        # patch-embed
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embedding_dim, norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # build-layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(
                dim = int(embedding_dim * 2 ** i_layer),
                depth = depths[i_layer],
                num_heads = num_heads[i_layer],
                window_size = window_size,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                drop_rate = drop_rate,
                norm_layer = norm_layer,
                downsample = PatchMerging if (i_layer < len(depths) - 1) else None
            )
            self.layers.append(layers)
        
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Perform swin transformer forward process.
        Args:
            x: torch.Tensor, [b, c, h, w]
        """
        x, h, w = self.patch_embedding(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, h, w = layer(x, h, w)
        
        x = self.norm(x)  # [b, l, c]
        x = self.avgpool(x.transpose(1, 2))  # [b, c, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x