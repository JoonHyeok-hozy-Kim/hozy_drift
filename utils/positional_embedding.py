import torch
import math

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)   # (e//2, )
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1).to(torch.float64)                     # (S, ) where S=num_patches
    out = torch.einsum('m,d->md', pos, omega).to(omega.dtype)   # (S, e//2)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)       # (S, e)


def get_sinusodial_pos_embed_num_patches(embed_dim, num_patches):
    pos = torch.arange(num_patches, dtype=torch.float32)            # (S, )
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)   # (S, e)
    return pos_embed.float()


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_height = torch.arange(grid_size, dtype=torch.float32)      # [0,...,h-1]
    grid_width = torch.arange(grid_size, dtype=torch.float32)       # [0,...,w-1]
    grid = torch.meshgrid(grid_width, grid_height, indexing='xy')   # [([0,0],...,[w-1,w-1]), ([0,0],...,[h-1,h-1])]
    grid = torch.stack(grid, dim=0)                                 # (2, h, w)
    
    emb_height = get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[1])   # (h*w, e//2)
    emb_width = get_1d_sincos_pos_embed_from_grid(embed_dim//2, grid[0])    # (h*w, e//2)
    return torch.cat([emb_height, emb_width], dim=1)  # (h*w, e)

def get_sinusodial_pos_embed_squared_img(embed_dim, num_patches):
    grid_size = int(math.sqrt(num_patches))
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)   # (S, e) where S=h*w
    return pos_embed.float()