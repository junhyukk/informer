import torch
import torch.nn as nn
from einops import rearrange, repeat
from compressai.entropy_models import EntropyBottleneck
from compressai.models.priors import JointAutoregressiveHierarchicalPriors
from torch.nn import functional as F
from compressai.models.utils import update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder
import warnings


class Img2Seq(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, attn_window=None):
        if attn_window is None:
            x_size = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')
            return x, x_size
        else:
            if x.shape[2] % attn_window != 0 or x.shape[3] % attn_window != 0:
                pad_h = attn_window - x.shape[2] % attn_window
                pad_w = attn_window - x.shape[3] % attn_window

                pad = (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2)
                x = F.pad(x, pad, "constant", 0)
            else:
                pad = None
            x_size = x.shape
            x = rearrange(x, 'b c (h ah) (w aw) -> (b h w) (ah aw) c',
                        ah=attn_window, aw=attn_window)
            return x, x_size, pad


class Seq2Img(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_size=None, attn_window=None, pad=None):
        if attn_window is None:
            x = rearrange(x, 'b (h w) c -> b c h w', h=x_size[2], w=x_size[3])
            return x
        else:
            x = rearrange(x, '(b h w) (ah aw) c -> b c (h ah) (w aw)',
                        b=x_size[0], h=x_size[2] // attn_window,
                        w=x_size[3] // attn_window, ah=attn_window,
                        aw=attn_window)
            if pad is not None:
                x = x[:, :, pad[2]:-pad[3], pad[0]:-pad[1]]
            return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, y):
        B, M, C = x.shape
        q = self.q(x).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        _, N, _ = y.shape
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, #heads, M, C) x (B, #heads, C, N) => (B, #heads, M, N) 
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, M, C)
        x = self.proj(x)
        return x


class CABlock(nn.Module):
    """ Transformer block using CrossAttention """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm_mlp = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, y):
        x = x + self.cross_attn(self.norm_q(x), self.norm_kv(y))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class Informer(JointAutoregressiveHierarchicalPriors):
    r""" Informer combined with the transformation part of [Minnen et al., NeurIPS 2018].

    [Minnen et al., NeurIPS 2018] 
    : Joint autoregressive and hierarchical priors for learned image compression 

    Args:
        N (int): Number of channels
        M (int): Number of channels in the last layer of the encoder
        num_global (int): Number of global tokens in the global hyperprior model 
    """
    
    def __init__(self, N=192, M=192, num_global=8, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        
        self.h_a = None
        self.h_s = None 
        self.entropy_bottleneck = None

        # Local hyeprprior model 
        self.local_h_a = nn.Sequential(
            nn.Conv2d(M, M // 4, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M // 4, M // 16, 1),
        )

        self.local_h_s = nn.Sequential(
            nn.Conv2d(M // 16, M // 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M // 2, M * 2, 1),
        )

        self.local_entropy_bottleneck = EntropyBottleneck(M // 16)

        # Global hyperprior model 
        self.num_global = num_global
        self.img2seq = Img2Seq()
        self.seq2img = Seq2Img()

        self.global_tokens = nn.Parameter(torch.randn(num_global, M))
        self.ca_a = CABlock(dim=M, num_heads=M // 64, qkv_bias=True)
        self.global_h_a = nn.Linear(M, M // num_global)
        self.global_h_s = nn.Linear(M // num_global, M * 2)

        self.global_entropy_bottleneck = EntropyBottleneck(M)

        # Parameter model 
        self.ca_s = CABlock(dim=M * 2, num_heads=M // 64 * 2, qkv_bias=True)

    def forward(self, x):
        # Transformation (encoder) + Quantization
        y = self.g_a(x)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        # Local hyperprior model
        local_z = self.local_h_a(y)
        local_z_hat, local_z_likelihoods = self.local_entropy_bottleneck(local_z)
        local_params = self.local_h_s(local_z_hat)

        # Global hyperprior model 
        global_seq = repeat(self.global_tokens, 'n c -> b n c', b=y.shape[0])
        y_seq, _ = self.img2seq(y)
        global_seq = self.ca_a(global_seq, y_seq)
        global_seq = self.global_h_a(global_seq)
        global_seq = rearrange(global_seq, 'b n c -> b (n c)')
        global_z = global_seq.unsqueeze(2).unsqueeze(3) # (B, M, 1, 1)
        global_z_hat, global_z_likelihoods = self.global_entropy_bottleneck(global_z)

        global_z_hat = global_z_hat.squeeze(3).squeeze(2)
        global_z_hat_seq = rearrange(global_z_hat, 'b (n c) -> b n c', n=self.num_global)
        global_params_seq = self.global_h_s(global_z_hat_seq)

        # Context model
        ctx_params = self.context_prediction(y_hat)

        # Parameter model 
        ctx_params_seq, ctx_params_size = self.img2seq(ctx_params)
        ctx_params_seq = self.ca_s(ctx_params_seq, global_params_seq)
        ctx_params = self.seq2img(ctx_params_seq, ctx_params_size)

        gaussian_params = self.entropy_parameters(
            torch.cat((local_params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        # Transformation (decoder)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "l_z": local_z_likelihoods, "g_z": global_z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.local_entropy_bottleneck,
            "local_entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.global_entropy_bottleneck,
            "global_entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        num_global = state_dict["global_tokens"].size(0)
        net = cls(N, M, num_global)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        local_z = self.local_h_a(y)
        local_z_strings = self.local_entropy_bottleneck.compress(local_z)
        local_z_hat = self.local_entropy_bottleneck.decompress(local_z_strings, local_z.size()[-2:])
        local_params = self.local_h_s(local_z_hat)

        global_seq = repeat(self.global_tokens, 'n c -> b n c', b=y.shape[0])
        y_seq, _ = self.img2seq(y)
        global_seq = self.ca_a(global_seq, y_seq)
        global_seq = self.global_h_a(global_seq)
        global_seq = rearrange(global_seq, 'b n c -> b (n c)')
        global_z = global_seq.unsqueeze(2).unsqueeze(3) # (B, M, 1, 1)
        global_z_strings = self.global_entropy_bottleneck.compress(global_z)
        global_z_hat = self.global_entropy_bottleneck.decompress(global_z_strings, global_z.size()[-2:])

        s = 1  # scaling factor between local_z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        y_height = local_z_hat.size(2) * s
        y_width = local_z_hat.size(3) * s
        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                local_params[i : i + 1],
                global_z_hat[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)
        return {"strings": [y_strings, local_z_strings, global_z_strings], "shape": local_z.size()[-2:]}

    def _compress_ar(self, y_hat, local_params, global_z_hat, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        global_z_hat = global_z_hat.squeeze(3).squeeze(2)
        global_z_hat_seq = rearrange(global_z_hat, 'b (n c) -> b n c', n=self.num_global)
        global_params_seq = self.global_h_s(global_z_hat_seq)

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                local_p = local_params[:, :, h : h + 1, w : w + 1]

                # Parameter model
                ctx_p_seq, ctx_p_size = self.img2seq(ctx_p)
                ctx_p_seq = self.ca_s(ctx_p_seq, global_params_seq)
                ctx_p = self.seq2img(ctx_p_seq, ctx_p_size)

                gaussian_params = self.entropy_parameters(torch.cat((local_p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 3

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        local_z_hat = self.local_entropy_bottleneck.decompress(strings[1], shape)
        local_params = self.local_h_s(local_z_hat)

        global_z_hat = self.global_entropy_bottleneck.decompress(strings[2], (1, 1))

        s = 1  # scaling factor between local_z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        y_height = local_z_hat.size(2) * s
        y_width = local_z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (local_z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=local_z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                local_params[i : i + 1],
                global_z_hat[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, local_params, global_z_hat, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        global_z_hat = global_z_hat.squeeze(3).squeeze(2)
        global_z_hat_seq = rearrange(global_z_hat, 'b (n c) -> b n c', n=self.num_global)
        global_params_seq = self.global_h_s(global_z_hat_seq)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                local_p = local_params[:, :, h : h + 1, w : w + 1]

                # Parameter model 
                ctx_p_seq, ctx_p_size = self.img2seq(ctx_p)
                ctx_p_seq = self.ca_s(ctx_p_seq, global_params_seq)
                ctx_p = self.seq2img(ctx_p_seq, ctx_p_size)

                gaussian_params = self.entropy_parameters(torch.cat((local_p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv