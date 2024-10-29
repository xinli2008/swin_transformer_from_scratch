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
