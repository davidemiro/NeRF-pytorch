import nerf.model
from utils.load_data import load_blender_data
from utils.configs_parser import configs_parser
from utils.rays_utils import *
from nerf.create_nerf import init_nerf
from nerf.rendering import render
import utils.rays_utils as rays_utils
import numpy as np
import torch
import time




def train():

    parser = configs_parser()
    args = parser.parse_args()

    #load lego data (blender data)
    images, poses, render_poses, hwf, index_split = load_blender_data("data/nerf_synthetic/lego")
    index_train, index_val, index_test = index_split

    H, W, focal = hwf

    near = 2.
    far = 6.

    if args.white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]


    model = nerf.model.MLP()

    model_fine = nerf.model.MLP()

    encodings_points = nerf.encoding.PositionalEncoder(args.L ,log_sampling=True, include_input=True, embed=True)

    encodings_view = nerf.encoding.PositionalEncoder(args.L_view, log_sampling=True, include_input=True, embed=True)



    # Create optimizer
    learning_rate = args.learning_rate
    if args.learning_rate_decay > 0:
        learning_rate = torch.ExponentialDecay(args.learning_rate,decay_steps=args.learning_rate_decay * 1000, decay_rate=0.1)
    optimizer = torch.optimizers.Adam(learning_rate)

    if args.batching:
        rays = [get_rays(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [N, ray origins + ray directions, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1) # [N, H, W, ray origins + ray directions + rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i] for i in index_train], axis=0)  # train images only # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        index_batch = 0


    for i in range(args.start, args.num_iterations):
        time0 = time.time()

        if args.batching:
            # Random over all images
            batch = rays_rgb[index_batch:index_batch+args.num_rand_rays]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, [1, 0, 2])

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2]

            index_batch += args.num_rand_rays
            if index_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                index_batch = 0







