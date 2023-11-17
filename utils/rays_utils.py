import torch
import numpy as np

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    c2w = torch.Tensor(c2w)
    i, j = torch.meshgrid(torch.arange(start=0,end=W, dtype=torch.float32),
                       torch.arange(start=0,end=H, dtype=torch.float32), indexing='xy')
    views = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(views[..., None, :] * c2w[:3, :3], -1)
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)
    return rays_o, rays_d