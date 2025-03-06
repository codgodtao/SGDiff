#################################################################################
#                                 MOE_Conv                                 #
#################################################################################
import torch
import math
from torch import autograd, nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


def get_device(x):
    gpu_idx = x.get_device()
    return f"cuda:{gpu_idx}" if gpu_idx >= 0 else "cpu"


class Router(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(in_channels, out_channels * 4),
            nn.LayerNorm(out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels)
        )
    def forward(self, x):
        return self.router(x)



class MoEConv(nn.Module):
    """"
    input: x (B C N H W) N代表图像通道，C代表channel，其实是卷积操作的专家数量
    output: y (B C N H W
    """

    def __init__(self, in_channels,out_channels, hidden_size, kernel_size=3, stride=1, padding=1,share_expert_ratio=0.1):
        super().__init__()
        self.share_expert = int(share_expert_ratio*in_channels)
        self.router1 = Router(hidden_size, in_channels)#out_channels is expert
        self.router2 = Router(hidden_size, in_channels)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()
        # self.shareExpert = nn.Conv3d(share_expert, share_expert, kernel_size, stride, padding)
        self.moeLayer1 = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding)
        self.moeLayer2 = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding)

    def forward(self, x, y1, y2): #scene grounding
        # 1.embedding-> softmax -> linear -> relu
        score1 = self.relu(self.router1(self.softmax(y1))) # batch_size hidden_size
        score2 = self.relu(self.router2(self.softmax(y2))) # batch_size hidden_size

        score1_fixed = score1.clone()
        score1_fixed[:, :self.share_expert] = 1.0

        score2_fixed = score2.clone()
        score2_fixed[:, :self.share_expert] = 1.0

        # 路由权重应用
        y1 = x * score1_fixed.view(*score1_fixed.shape, 1, 1, 1)
        y1 = self.relu(self.moeLayer1(y1))

        y2 = y1 * score2_fixed.view(*score2_fixed.shape, 1, 1, 1)
        y2 = self.moeLayer2(y2)

        return y2+y1, torch.abs(score1).mean(), torch.abs(score2).mean()



class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

# class Downsample(nn.Module):
#     def __init__(self, n_feat):
#         super(Downsample, self).__init__()
#         self.body = nn.Conv3d(n_feat, 2*n_feat, kernel_size=3, stride=1, padding=1, bias=False)
#
#     def forward(self, x):
#         # b, n, c, h, w = x.size()
#         x = F.interpolate(x, scale_factor=(1,0.5,0.5), mode='trilinear') #->  b 2c n h//2 w//2
#         x = self.body(x)
#         # x = x.transpose(1, 2).reshape(b * c, n // 2, h, w)
#         # x = self.unshuffle(x).reshape(b, c, n * 2, h // 2, w // 2).transpose(2, 1)  #b, 2*n, c, h//2, w//2
#         return x
#
#
# class Upsample(nn.Module):
#     def __init__(self, n_feat):
#         super(Upsample, self).__init__()
#         self.body = nn.Conv3d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False)
#
#     def forward(self, x):
#         # b, c, n, h, w = x.size()  ->  b c//2 n h*2 w*2
#         x = F.interpolate(x, scale_factor=(1,2,2), mode='trilinear')
#         x = self.body(x)
#         return x

def to_3d(x):
    return rearrange(x, 'b c n h w -> b (n h w) c')


def to_4d(x, n, h, w):
    return rearrange(x, 'b (n h w) c -> b c n h w', n=n, h=h, w=w)


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm3d(self.half, affine=True)
        self.BN = nn.BatchNorm3d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="IBN"):
        super(LayerNorm, self).__init__()
        self.LayerNorm_type = LayerNorm_type
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        elif LayerNorm_type == "IBN":
            self.body = IBN(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if self.LayerNorm_type == 'IBN':
            return self.body(x)
        n, h, w = x.shape[-3:]
        return to_4d(self.body(to_3d(x)), n, h, w)  # 现在B C (NHW)内部norm然后转为b c n h w


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
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

        out = self.project_out(out)
        return out


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None, None]

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention3D(nn.Module):
    def __init__(self, dim, num_heads, bias, hidden_size):
        super(Attention3D, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.dense1 = Dense(hidden_size, dim)
        self.dense2 = Dense(hidden_size, dim)
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, t):
        b, n, c, h, w = x.shape
        x = x * self.dense1(t) + self.dense2(t)
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head n) c h w -> b head n (c h w)', head=self.num_heads)  # n*h*w就是 hidden dim
        k = rearrange(k, 'b (head n) c h w -> b head n (c h w)', head=self.num_heads)
        v = rearrange(v, 'b (head n) c h w -> b head n (c h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)  # b head c*c的attention score

        out = (attn @ v)

        out = rearrange(out, 'b head n (c h w) -> b (head n) c h w', head=self.num_heads, c=c, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class Transformer3DBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type, hidden_size):
        super(Transformer3DBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention3D(dim, num_heads, bias, hidden_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = MoEConv(in_channels=dim, out_channels=dim, hidden_size=hidden_size)

    def forward(self, x, t, y1, y2):
        x = x + self.attn(self.norm1(x), t)
        x, l1_loss, div_loss = self.ffn(self.norm2(x), y1, y2) #采用residual会让模型偷懒，直接跳过这部分FFN
        return x, l1_loss,div_loss


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Conv3d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        #b c h w-> b c//2 h w-> b 2c h//2 w//2  2D阶段的DownSample

        #b n c h w-> b n//2 c h w-> b 2n c h//2 w//2
        b, n, c, h, w = x.size()
        x = self.body(x) #b, n//2, c, h, w
        x = x.transpose(1, 2).reshape(b * c, n // 2, h, w)
        x = self.unshuffle(x).reshape(b, c, n * 2, h // 2, w // 2).transpose(2, 1)  #b, 2*n, c, h//2, w//2
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Conv3d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        b, c, n, h, w = x.size()
        x = self.body(x)
        x = x.transpose(1, 2).reshape(b * n, c * 2, h, w)
        x = self.shuffle(x).reshape(b, n, c // 2, h * 2, w * 2).transpose(2, 1)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size * 4),  # 不同结构
            nn.LayerNorm(hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).type(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


dtype = torch.float32


class LanguageEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    #这里今天下午刚刚做了修改
    """

    def __init__(self, hidden_size, model_clip, language_embedding=768):
        super().__init__()
        #hidden_size=256
        self.model_clip = model_clip.eval()
        self.mlp_prompt1 = nn.Sequential(
            nn.Linear(language_embedding, hidden_size * 4),  # 不同结构
            nn.LayerNorm(hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.mlp_prompt2 = nn.Sequential(
            nn.Linear(language_embedding, hidden_size * 4),  # 不同结构
            nn.LayerNorm(hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature.type(dtype)

    def forward(self, prompts1, prompts2):
        t_emb = self.get_text_feature(prompts1)
        t_emb2 = self.get_text_feature(prompts2)
        t_emb = self.mlp_prompt1(t_emb)
        t_emb2 = self.mlp_prompt2(t_emb2)
        return t_emb, t_emb2

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out,hidden_size):
        super().__init__()
        "(b,32,h,w,c)->(b,32,h,w,c)"

        self.output = nn.Conv3d(channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x,t):  # norm silu dropout conv为一个block单元

        return self.output(x)

##########################################################################
##---------- Restormer -----------------------
class MOENetWork(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 hidden_size=256,#router
                 dim=32,
                 heads=None,
                 bias=False,
                 model_clip=None,
                 LayerNorm_type='BiasFree'
                 ):

        super(MOENetWork, self).__init__()

        if heads is None:
            heads = [1, 2, 4, 8] #head_dim attention中的dim, c c**2 c**2**2 c**2**2**2
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.l_embedder = LanguageEmbedder(hidden_size, model_clip)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = Transformer3DBlock(dim=dim, num_heads=heads[0], bias=bias, LayerNorm_type=LayerNorm_type,
                                                 hidden_size=hidden_size)

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = Transformer3DBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], bias=bias,
                                                 LayerNorm_type=LayerNorm_type, hidden_size=hidden_size)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3  *2
        self.encoder_level3 = Transformer3DBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                                 bias=bias, LayerNorm_type=LayerNorm_type, hidden_size=hidden_size)

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4  *4
        self.latent = Transformer3DBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                                         bias=bias, LayerNorm_type=LayerNorm_type, hidden_size=hidden_size)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3      *8
        self.reduce_chan_level3 = nn.Conv3d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = Transformer3DBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                                 bias=bias, LayerNorm_type=LayerNorm_type, hidden_size=hidden_size)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = Transformer3DBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                                 bias=bias, LayerNorm_type=LayerNorm_type, hidden_size=hidden_size)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv3d(int(dim * 2 ** 1), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = Transformer3DBlock(dim=dim, num_heads=heads[0],
                                                 bias=bias, LayerNorm_type=LayerNorm_type, hidden_size=hidden_size)

        self.refinement = Transformer3DBlock(dim=int(dim), num_heads=heads[0],
                                             bias=bias, LayerNorm_type=LayerNorm_type, hidden_size=hidden_size)

        self.output = ResBlock(dim,out_channels,hidden_size=hidden_size)
    def forward(self, inp_img, t, prompts1, prompts2): #scene grounding
        # b 3(xt,pan,ms) c h w input --> b 1 c h w (res)  for loss function
        t = self.t_embedder(t)  # (N, D)
        total_l1_loss = 0
        total_div_loss = 0
        y1, y2 = self.l_embedder(prompts1, prompts2)
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1, l1_loss, div_loss = self.encoder_level1(inp_enc_level1, t, y1, y2)
        total_l1_loss += l1_loss
        total_div_loss += div_loss
        # print(inp_enc_level1.shape, out_enc_level1.shape) # N C H W

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, l1_loss, div_loss = self.encoder_level2(inp_enc_level2, t, y1, y2)
        total_l1_loss += l1_loss
        total_div_loss += div_loss #不添加约束lOSS会变得完全一致！
        # print(inp_enc_level2.shape, out_enc_level2.shape) # 2N C H/2 W/2

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, l1_loss, div_loss = self.encoder_level3(inp_enc_level3, t, y1, y2)
        total_l1_loss += l1_loss
        total_div_loss += div_loss
        # print(inp_enc_level3.shape, out_enc_level3.shape)# 4N C H/4 W/4

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent, l1_loss, div_loss = self.latent(inp_enc_level4, t, y1, y2)
        total_l1_loss += l1_loss
        total_div_loss += div_loss
        # print(inp_enc_level4.shape, latent.shape)# 8N C H/8 W/8

        inp_dec_level3 = self.up4_3(latent)# 4N C H/4 W/4
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1) # 8N C H/4 W/4
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)# 4N C H/4 W/4
        out_dec_level3, l1_loss, div_loss = self.decoder_level3(inp_dec_level3, t, y1, y2)
        total_l1_loss += l1_loss
        total_div_loss += div_loss
        # print(inp_dec_level3.shape,out_dec_level3.shape)

        inp_dec_level2 = self.up3_2(out_dec_level3)# 2N C H/2 W/2
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)# 2N C H/2 W/2
        out_dec_level2, l1_loss, div_loss = self.decoder_level2(inp_dec_level2, t, y1, y2)
        total_l1_loss += l1_loss
        total_div_loss += div_loss
        # print(inp_dec_level2.shape, out_dec_level2.shape)

        inp_dec_level1 = self.up2_1(out_dec_level2)# N C H W
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1) # N C H W
        out_dec_level1, l1_loss, div_loss = self.decoder_level1(inp_dec_level1, t, y1, y2)# N C H W
        total_l1_loss += l1_loss
        total_div_loss += div_loss
        # print(inp_dec_level1.shape, out_dec_level1.shape)

        out_dec_level1 = out_dec_level1 + inp_enc_level1  # N*2
        out_dec_level1, l1_loss, div_loss = self.refinement(out_dec_level1, t, y1, y2)
        total_l1_loss += l1_loss
        total_div_loss += div_loss
        # print(out_dec_level1.shape)

        out_dec_level1 = self.output(out_dec_level1,t) #1
        # print(out_dec_level1.shape)
        return out_dec_level1, total_l1_loss, total_div_loss


# # Example usage
if __name__ == "__main__":
    import clip
    import numpy as np

    device = "cuda:3"
    model_clip, _ = clip.load("ViT-L/14", device=device)  #we need embeding for MOE conv layer
    y = "this is a test"
    y2 = "this is a test"
    y = clip.tokenize(y, truncate=True).to(device) #tokenizer->77 embeding
    y2 = clip.tokenize(y2, truncate=True).to(device)

    print(y.shape,y2.shape)
    time_in = torch.from_numpy(np.random.randint(1, 1000 + 1, size=1)).to(device)
    model = MOENetWork(inp_channels=3, out_channels=1, dim=16, model_clip=model_clip).to(device)
    img = torch.randn(10, 3, 8, 64, 64).to(device)
    output = model(img, time_in, y, y2)

    for name, param in model.named_parameters():
        if "router" in name or "l_embedder" in name:
            print(name)
            param.requires_grad = False

    # print(output.shape)
