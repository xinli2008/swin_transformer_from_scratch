import torch
# NOTE:设置pytorch的张量打印选项, 具体来说, 它将threshold参数设置为torch.inf, 意味着在打印tensor时, 不会对元素的数量进行截断。
# 这意味着无论tensor多大, 所有的元素都会被完整的展示出来。
# torch.set_printoptions(threshold=torch.inf)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class PatchEmbedding(nn.Module):
    """
    2D Image Patch Embedding
    """
    def __init__(self, patch_size = 4, in_channels = 3, embedding_dim = 96, norm_layer = None):
        super(PatchEmbedding, self).__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        # NOTE: 通过设置kernel_size和stride等于patch_size的卷积, 来完成patch_embedding。
        self.projection = nn.Conv2d(self.in_channels, self.embedding_dim, kernel_size = self.patch_size, stride = self.patch_size)
        self.norm = norm_layer(self.embedding_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        Perform patch embedding forward
        Args:
            x: torch.Tensor, [b, c, h, w]
        Return:
            x: [b, L, c]
        """
        _, _, h, w = x.shape

        # NOTE: 如果输入图片的宽高不是patch_size的整数倍, 需要padding操作。
        pad_input = (h % self.patch_size[0] != 0) or (w % self.patch_size[1] != 0)
        if pad_input:
            # 在图片的宽高维度进行padding
            # NOTE: pad是从最后一个维度依次向前padding, (w_left, w_right, h_top, h_bottom, c_front, c_back)
            # 所以以下的操作是在w_right和h_top维度上进行padding。
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
        super(PatchMerging, self).__init__()
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


def window_partition(x, window_size):
    r"""
    将feature_map按照window_size划分成没有重叠的window
    Args:
        x: torch.Tensor, [b, h, w, c]
        window_size: int
    Returns:
        windows: [b * number_window, window_size, window_size, c]
    """
    b, h, w, c = x.shape     
    # [b, h, w, c] -> [b, window_count_h, window_size, window_count_w, window_size, c]
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    # permute: [b, window_count_h, window_count_w, window_size, window_size, c]
    # view: [b * window_count_h * window_count_w, window_size, window_size, c]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    r"""
    将一个个不重叠的window还原成之前的特征图
    Args:
        windows: torch.Tensor, [b * number_window, window_size, window_size, c]
        window_size: int
        h: int
        w: int
    Returns:
        x: torch.Tensor, [b, h, w, c]
    """
    b = int(windows.shape[0] / ( h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer = nn.GELU, drop_rate = 0.1):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_rate)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

class WindowAttention(nn.Module):
    r"""window MSA and shift window MSA"""
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        assert dim % num_heads==0, "dim should be divisble by num_heads"
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # NOTE: 根据windows_size来定义一个相对位置编码的表格
        # 假设当window_size = 7时, 每一个位置与其他位置的相对关系一共有2*7-1=13种。
        # 所以这个relative_position_bias_table的形状应该是(2*window_size-1)
        self.realative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size- 1) * (2 * window_size - 1), num_heads)
        )

        # NOTE: 接下来, 在这个表格中生成相对位置索引(index)
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # NOTE: 这里是将二元索引变为一元索引的过程了:
        # 需要将行标加上window_size - 1, 列标也加上window_size - 1
        # 然后行标乘上2*windows_size - 1, 然后行标和列表相加就完成了。
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)

        # NOTE: 因为这个相对位置索引, 一旦计算好, 就不用改变并且不需要训练, 所以我们使用reister_buffer来将它放入内存中。
        self.register_buffer("relative_position_index", relative_position_index)

        self.w_q = nn.Linear(dim, dim, bias = False)
        self.w_k = nn.Linear(dim, dim, bias = False)
        self.w_v = nn.Linear(dim, dim, bias = False)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        Perform window/shifted window MSA
        Args:
            x: torch.Tensor, [b*number_count, number_window_h * num_window_w, c]
            mask: (0/-inf) torch.Tensor, [number_count, number_window_h*number_window_w, number_window_h*number_window_w] or None
        """
        batch, seq_length, dimension = x.shape
        q = self.w_q(x).reshape(batch, seq_length, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.w_k(x).reshape(batch, seq_length, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.w_v(x).reshape(batch, seq_length, self.num_heads, -1).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores * self.scale

        # NOTE: 接下来需要根据相对位置编码去相对位置表格中去获取元素
        relative_position_bias = self.realative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        # NOTE: 如果mask为None的话, 则
        if mask is not None:
            # mask: [number_count, window_size_h * window_size_w, window_size_h * window_size_w]
            num_window = mask.shape[0]
            attention_scores = attention_scores.view(batch // num_window, num_window, self.num_heads, seq_length, seq_length) + mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_heads, seq_length, seq_length)
            attention_scores = self.softmax(attention_scores)
        else:
            attention_scores = self.softmax(attention_scores)

        # 至此, 注意力机制中softmax和softmax括号内的运算已经完成
        x = torch.matmul(attention_scores, v).transpose(1, 2).reshape(batch, seq_length, dimension)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    swin transformer block
    """
    def __init__(self, dim, num_heads, windows_size, shift_size, mlp_ratio, qkv_bias, drop_rate, act_layer, norm_layer):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = windows_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size <= self.window_size, "shift_size has a wrong size!"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, windows_size, num_heads)
        self.dropout = nn.Dropout(drop_rate)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features = dim, hidden_features = mlp_hidden_dim, out_features = dim, act_layer = act_layer, drop_rate = drop_rate)
    
    def forward(self, x, attention_mask):
        h, w = self.h, self.w
        b, l, c = x.shape
        assert l == h * w, "input features has wrong size"

        shortcut = x
        x = self.norm1(x)
        # NOTE: 在之前写的vit-transformer-block中, 一直是以[b, L, c]的shape进行操作,
        # 而在swin的block中, 因为需要对窗口进行分区, 所以需要记录h,w, 然后先将[b, L, c]->[b, c, h, w]再进行操作。
        x = x.view(b, h, w, c)

        # NOTE:将特征图给padding成windows_size的整数倍, 方便于接下来将特征图给划分成不重叠的窗口
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # NOTE:进行窗口平移
        # 如果shift_size>0, 则需要将上面的某些行和左面的某些列进行移动。
        # 将上面的shift_size行移动到下方, 将左侧的shift_size列移动到右边去。
        if self.shift_size > 0:
            shift_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims = (1, 2))
        else:
            shift_x = x
            attention_mask = None

        # NOTE: 将特征图按窗口进行拆分
        x_windows = window_partition(shift_x, self.window_size)  # [b * number_count_w * number_count_h, window_size, window_size, c]
        # 在放入attn之前, 输入已经处理成[b, L, c]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # [b * number_count, window_size * window_size, c]

        # NOTE: 窗口/移动窗口自注意力机制
        attention_windows = self.attn(x_windows, mask = attention_mask)

        # NOTE: 合并窗口
        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attention_windows, self.window_size, Hp, Wp)

        # NOTE: 将经过移动的特征图回复原样
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts = (self.shift_size, self.shift_size), dims = (1,2))
        else:
            x = shift_x

        # NOTE: 将前面padding的数据给去掉
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()
        
        x = x.reshape(b, h*w, c)

        x = shortcut + self.dropout(x)
        x = x + self.mlp(self.norm2(x))

        return x
    

class BasicLayer(nn.Module):
    """
    a base swin transformer layer for each stage
    """
    def __init__(self, dim, depth, num_heads, windows_size, mlp_ratio, qkv_bias, drop_rate, norm_layer, downsample):
        super().__init__()
        self.dim = dim
        self.window_size = windows_size
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
                act_layer = nn.GELU, 
                norm_layer = norm_layer
            )
         for i in range(depth)   
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim = dim, norm_layer = norm_layer)
        else:
            self.downsample = None
    
    def create_mask(self, x, h, w):
        r"""
        生成移动窗口自注意力机制所需要的attention mask。
        这个mask的作用时在平移窗口中创建屏蔽区域, 从而更好的实现跨窗口信息交互。
        """
        # 计算Hp和Wp,确保它们是window_size的整数倍
        Hp = int(np.ceil(h / self.window_size)) * self.window_size
        Wp = int(np.ceil(w / self.window_size)) * self.window_size

        # 创建image_mask, 并且为每个区域创建编号, 编号相同的为同一区域。
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        # NOTE: 根据移动窗口-自注意力机制的图片, 我们可以看到通过window-partition的方式, 将一个图片分成了九个sub-element。
        # 而这九个部分从横轴和纵轴分别是以下的几个切片。
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # 将img_mask分割成窗口并重塑为二维张量
        mask_windows = window_partition(img_mask, self.window_size)  # [b * window_count_h * window_count_w, window_size, window_size, c]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # [b * window_count_h * window_count_w * c, window_size * window_size]
        
        # NOTE: 通过对窗口内的编号进行差值计算, 生成注意力掩码。差值不为0的位置表示不同的窗口区域。
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        # NOTE: 将不同的区域的位置填充为-100, 表示在自注意力中抑制这些位置的影响;
        # 将相同区域的位置填充为0, 表示正常的注意力。
        attention_mask = attention_mask.masked_fill(attention_mask!=0, float(-100.0)).masked_fill(attention_mask==0, float(0.0))
        
        return attention_mask

    def forward(self, x, h, w):
        attention_mask = self.create_mask(x, h, w)
        
        for block in self.blocks:
            block.h, block.w = h, w
            x = block(x, attention_mask)
        
        # NOTE: 经过每一个stage后, 后面会接一个patch_merging操作来降低图像的分辨率
        if self.downsample is not None:
            x = self.downsample(x, h, w)
            h, w = (h + 1) // 2, (w + 1) // 2
        
        return x, h, w


class SwinTransformerTiny(nn.Module):
    """
    Swin Transformer Tiny Version
    """
    def __init__(self, patch_size = 4, in_channels = 3, num_classes = 1000,
                embedding_dim = 96, depths = (2, 2, 6, 2), num_heads = (3, 6, 12, 24),
                window_size = 7, mlp_ratio = 4, qkv_bias = True, drop_rate = 0.1, 
                norm_layer = nn.LayerNorm, patch_norm = True):
        super(SwinTransformerTiny, self).__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embedding_dim = embedding_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        # NOTE: 每经过一个stage, 通过patch_merding的操作, num_features就会变大两倍。
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
                windows_size = window_size,
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
        # self.apply(self._init_weights)
    
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
        x, h, w = self.patch_embedding(x)   # [b, c, h, w] -> [b, l, c]
        x = self.pos_drop(x)                # [b, l, c]

        for layer in self.layers:
            x, h, w = layer(x, h, w)
        
        x = self.norm(x)                    # [b, l, c]
        x = self.avgpool(x.transpose(1, 2)) # [b, c, 1]
        x = torch.flatten(x, 1)             # [b, c]
        x = self.head(x)                    # [b, 1]
        return x


if __name__ == "__main__":
    input_tensor = torch.randn([2, 3, 224, 224])
    model = SwinTransformerTiny()
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
