from utils.load_data import load_blender_data
from utils.configs_parser import configs_parser
from nerf.create_nerf import init_nerf
from nerf.rendering import render
import utils.rays_utils as rays_utils
import numpy as np
import torch




def train():

    parser = configs_parser()
    args = parser.parse_args()

    #load lego data
    images, poses, render_poses, hwf, index_split = load_blender_data("data/nerf_synthetic/lego")
    index_train, index_val, index_test = index_split

    near = 2.
    far = 6.

    if args.white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]

    #Define Width W Height H and Focal length focal
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    num_iterations = 100000
    for i in range(num_iterations):

        # Random from one image
        index = np.random.choice(index_train)
        target = images[index]
        pose = poses[index, :3, :4]

        #get the info of each ray in the image
        rays_origin, rays_direction = rays_utils.get_rays(H, W, focal, pose)

        coords = torch.stack(torch.meshgrid(
                torch.arange(0,H), torch.arange(0,W), indexing='ij'), -1)
        coords = torch.reshape(coords, [-1, 2]) # (H * W, 2)
        select_inds = np.random.choice(
            coords.shape[0], size=[args.num_rand], replace=False) # (num_rand,)

        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_origin = rays_origin[select_coords[:,0],select_coords[:,1]] # (num_rand, 3)
        rays_direction = rays_direction[select_coords[:,0],select_coords[:,1]] # (num_rand,3)
        batch_rays = torch.stack([rays_origin, rays_direction], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]


        models = init_nerf()
        rgb, disp, acc, extras = render(H, W, focal, models, args.num_coarse_samples, args.num_fine_samples, chunk=args.num_chunk, rays=batch_rays)



train()


