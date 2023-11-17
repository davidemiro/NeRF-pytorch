import torch
import torch.nn as nn
import collections
from nerf.sampling import sample_pdf
from nerf.create_nerf import init_nerf


def batchify_rays(rays_flat,models, num_coarse_samples,num_fine_samples, chunk=1024*32):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk],models,num_coarse_samples,num_fine_samples)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal,models,num_coarse_samples,num_fine_samples,chunk=1024*32, rays=None,near=0., far=1.):
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      models: coarse_mlp, fine_mlp, pts positional encodings and views positional encodings.
      num_coarse_samples: number of samples that corase_mlp accepts as input.
      num_fine_samples: number of samples that fine_mlp accepts as input.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """


    # use provided ray batch
    rays_origin, rays_directions = rays

    # provide ray directions as input
    views = rays_directions

    # Make all directions unit magnitude.
    # shape: [batch_size, 3]
    views = views / torch.norm(views, dim=-1, keepdim=True)
    views = torch.reshape(views, [-1, 3]).type(torch.float32)

    sh = rays_directions.shape  # [..., 3]


    # Create ray batch
    rays_origin = torch.reshape(rays_origin, [-1, 3]).type(torch.float32)
    rays_direction = torch.reshape(rays_directions, [-1, 3]).type(torch.float32)
    near, far = near * torch.ones_like(rays_directions[..., :1]), far * torch.ones_like(rays_direction[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = torch.cat([rays_origin, rays_direction, near, far], axis=-1)

    # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
    rays = torch.cat([rays, views], axis=-1)

    # Render and reshape
    all_ret = batchify_rays(rays,models,num_coarse_samples,num_fine_samples, chunk)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_rays(ray_batch,
                models,
                num_coarse_samples,
                num_fine_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                num_importance=0):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      models: coarse_mlp, fine_mlp, points positional_embedding and views positional_embeddings
      num_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      num_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    #initialize model
    coarse_mlp, fine_mlp, embedding_pts, embedding_views = models

    # batch size
    num_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_origins, rays_directions = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    views = ray_batch[:, -3:]



    # Extract lower, upper bound for ray distance.
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = torch.linspace(0., 1., num_coarse_samples)

    z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = torch.broadcast_to(z_vals, [num_rays, num_coarse_samples])



    # Points in space to evaluate model at.
    pts = rays_origins[..., None, :] + rays_directions[..., None, :] * z_vals[..., :, None]  # [num_rays, num_samples, 3]


    views = views[:, None].expand(pts.shape) #[ num_rays, num_samples,3]



    # Points and views embeddings
    pts = embedding_pts(pts)
    views = embedding_views(views)

    # Evaluate model at each point
    raw = coarse_mlp(pts,views) # [num_rays, num_samples, 4]

    # Apply volume rendering function
    rgb_map, disp_map, acc_map, weights, depth_map = volume_rendering_function(raw, z_vals, rays_directions,0)

    num_fine_samples = 64
    if num_fine_samples > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        #stratified sampling

        # Make predictions with mlp_fine.
        pts = embedding_pts(pts)
        views = embedding_views(views)

        # Evaluate model at each point
        raw = fine_mlp(pts, views)  # [num_rays, num_samples, 4]

        rgb_map, disp_map, acc_map, weights, depth_map = volume_rendering_function(raw, z_vals, rays_directions)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if num_fine_samples > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = z_samples.std(-1,unbiased=True)  # [num_rays]

    return ret


def raw_to_alpha(raw, deltas, act_fn=nn.ReLU()):
    return 1.0 - torch.exp(-act_fn(raw) * deltas)

def cumprod_exclusive(x):
    """

    implementation of the tf.math.cumprod(...,exclusive=True)
    prod_i = 1 * elem_1 * .... * elem_i-1
    :param x: Tensor
    :return:
    """
    return torch.cumprod(x) / x

def volume_rendering_function(raw, t, d, raw_noise_std):
    """

    :param raw: [num_rays,num_samples_x_ray,4] output of the model (r,g,b,σ)
    :param t: [num_rays, num_samples_x_ray]
    :param d: [num_ray,3] direction of each ray
    :return:
    """

    # computes delta_i
    deltas = t[...,1:] - t[...,:-1]

    # inf - delta_N-1
    deltas = torch.cat([deltas, torch.broadcast_to(torch.Tensor([1e10]), deltas[..., :1].shape)], dim=-1)

    deltas = deltas * torch.norm(d[..., None, :], dim=-1)

    # Extract RGB of each sample position along each ray.
    rgb = torch.sigmoid(raw[..., :3])  # [num_rays, num_samples, 3]

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.normal(raw[..., 3].shape) * raw_noise_std

    # compute alphas along each ray
    alphas = raw_to_alpha(raw[...,:3]+noise,deltas)

    # Compute weight for RGB of each sample along each ray.
    weights = alphas * cumprod_exclusive(1. - alphas + 1e-10, dim=-1) #[num_rays, num_samples]

    # Computed weighted color of each sample along each ray.
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [num_rays, 3]

    # Estimated depth map is expected distance.
    depth_map = torch.sum(weights * t, dim=-1)

    # Disparity map is inverse depth.
    disp_map = 1. / torch.maximum(1e-10, depth_map /torch.sum(weights, dim=-1))

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map.
    rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map
















