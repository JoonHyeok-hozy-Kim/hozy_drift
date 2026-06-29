import torch
import torch.nn as nn

from models.attention import Transformer
from utils.positional_embedding import get_sinusodial_pos_embed_num_patches, \
    get_sinusodial_pos_embed_squared_img

# For adaLN 
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class PushForwardBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        num_patches,
        attn_num_heads,
        ffn_expansion,
    ):
        super().__init__()
        assert in_channels % attn_num_heads == 0, \
            f"attn_num_heads(={attn_num_heads}) doesn't divide in_channels(={in_channels})."
            
        self.in_channels = in_channels
        self.num_patches = num_patches        
        attn_head_dim = in_channels // attn_num_heads
        self.transformer = Transformer(in_channels, attn_num_heads, attn_head_dim, ffn_expansion)
        
        self._init_linear_weights()
    
    def _init_linear_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y, attn_temp=1.0):
        if y is not None:
            raise NotImplementedError("Conditional input case is not implemented yet.")
            # [hozy] Label Embedder Implementation required.
            # [hozy] adaLN Implementation required.
                
        x = self.transformer(x, attn_mask=None, attn_temp=attn_temp, cache_key=None)
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.layer_norm = nn.LayerNorm(in_channels, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(in_channels, out_channels, bias=True)        
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, y):
        if y is not None:
            raise NotImplementedError("Conditional input case is not implemented yet.")
            # [hozy] adaLN implementation required.
            
        x = self.layer_norm(x)
        x = self.linear(x)                
        return x



class DriftModelRaw(nn.Module):
    def __init__(
        self, 
        in_channels,    # C
        num_patches,    # S
        num_blocks,
        hidden_dim,     # D
        attn_num_heads,
        ffn_expansion,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_patches = num_patches
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        
        # # Sinusoidal Positional Embedding
        # pos_embed = get_sinusodial_pos_embed_num_patches(hidden_dim, num_patches)    # (S, D)
        # pos_embed = pos_embed.unsqueeze(0)                                           # (1, S, D)
        # self.register_buffer("pos_embed", pos_embed)
        
        # Simple Linear Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(num_patches, hidden_dim) * 1e-2)
        
        self.w_in = nn.Linear(in_channels, hidden_dim, bias=True)
        nn.init.xavier_uniform_(self.w_in.weight)
        nn.init.zeros_(self.w_in.bias)        
        
        self.push_forward_blocks = nn.ModuleList([
            PushForwardBlock(hidden_dim, num_patches, attn_num_heads, ffn_expansion) \
                for _ in range(self.num_blocks)
        ])        
        self.final_layer = FinalLayer(hidden_dim, self.out_channels)

    def forward(self, x, y=None):
        x = self.w_in(x)                    # (B, S, D)
        x = x + self.pos_embed.to(x.dtype)  # (B, S, D) 
        
        for push_forward in self.push_forward_blocks:
            x = push_forward(x, y)            
        x = self.final_layer(x, y)
        
        return x



class DriftModelImage(DriftModelRaw):
    def __init__(
        self, 
        in_channels,    # C
        num_patches,    # S
        num_blocks,
        hidden_dim,     # D
        attn_num_heads,
        ffn_expansion,
    ):
        super().__init__(
            in_channels,    # C
            num_patches,    # S
            num_blocks,
            hidden_dim,     # D
            attn_num_heads,
            ffn_expansion,
        )
        
        pos_embed = get_sinusodial_pos_embed_squared_img(hidden_dim, num_patches)    # (S, D)
        pos_embed = pos_embed.unsqueeze(0)                                           # (1, S, D)
        self.register_buffer("pos_embed", pos_embed)
        
    
    def patchify(self, x):
        B, C, H, W = x.shape
        S = self.patch_size
        h, w = H//S, W//S
        x = x.reshape(B, C, h, S, w, S)
        x = x.permute(0, 2, 4, 1, 3, 5)     # (B, h, w, C, S, S)
        x = x.reshape(B, h*w, C*S*S)        # (B, T, c) 
        return x
    
    
    def unpatchify(self, x):
        B, _, c = x.shape
        S = self.patch_size
        h = w = self.img_size // S
        C = c // (S**2)
        x = x.reshape(B, h, w, C, S, S)
        x = x.permute(0, 3, 1, 4, 2, 5)     # (B, C, h, S, w, S)
        x = x.reshape(B, C, h*S, w*S)       # (B, C, H, W)
        return x        
    
    def forward(self, x, y=None):
        x = self.patchify(x)                # (B, T, c)     
        x = self.w_in(x)                    # (B, T, D)
        x = x + self.pos_embed.to(x.dtype)  # (B, T, D) where T=S for squared images
        
        for push_forward in self.push_forward_blocks:
            x = push_forward(x, y)            
        x = self.final_layer(x, y)
        
        x = self.unpatchify(x)
        
        return x