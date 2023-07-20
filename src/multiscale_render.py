import os
import itertools
import sys
from copy import copy
from typing import List, Dict, Any, Optional, NamedTuple, TypedDict
from pathlib import Path

import imageio
import torch
import numpy as np
from tqdm import tqdm
from torch.distributions.bernoulli import Bernoulli

from run_nerf import raw2outputs
from run_nerf_helpers import to8b, get_rays, get_rays_np, mse2psnr, img2mse
from utils import Logger, ConfigManager, PerfMonitor, has_flag, load_yaml_as_dict, parse_args_and_init_logger
from load_nsvf_dataset import load_nsvf_dataset
from local_distill import create_multi_network_fourier_embedding, create_multi_network

import kilonerf_cuda

import rerun as rr

device = torch.device("cuda")


LOGGING_TRANSFORM = np.diag([1, -1, -1, 1])
USE_RR = False


def get_points_in_occupied_space(points_flat, occupancy_grid, cfg):
    global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(points_flat.device)
    global_domain_size = global_domain_max - global_domain_min

    res = cfg['occupancy']['resolution']
    occupancy_resolution = torch.tensor(res, dtype=torch.long, device=points_flat.device)
    strides = torch.tensor([res[2] * res[1], res[2], 1], dtype=torch.long,
                           device=points_flat.device)  # assumes row major ordering
    voxel_size = global_domain_size / occupancy_resolution
    occupancy_indices = ((points_flat - global_domain_min) / voxel_size).to(torch.long)
    torch.max(torch.tensor([0, 0, 0], device=points_flat.device), occupancy_indices, out=occupancy_indices)
    torch.min(occupancy_resolution - 1, occupancy_indices, out=occupancy_indices)
    occupancy_indices = (occupancy_indices * strides).sum(dim=1)

    point_in_occupied_space = occupancy_grid[occupancy_indices]

    return point_in_occupied_space

# TODO
def query_multi_network(
        networks,
        points,
        directions,
        occupancy_grid,
        cfg,
        distance_mask,
        resolution,
        freq_index,
        **kwargs,
):
    num_rays = points.size(0)
    num_samples = points.size(1)
    points_flat = points.view(-1, 3)

    multi_network = networks[freq_index][resolution]["model"]
    res = [resolution, resolution, resolution]

    position_fourier_embedding = networks[freq_index][resolution]["position_fourier_embedding"]
    direction_fourier_embedding = networks[freq_index][resolution]["direction_fourier_embedding"]

    domain_mins = networks[freq_index][resolution]["domain_mins"]
    domain_maxs = networks[freq_index][resolution]["domain_maxs"]
    debug_network_color_map = networks[freq_index][resolution]["debug_network_color_map"]

    # res = cfg['fixed_resolution']
    num_networks = multi_network.num_networks
    fixed_resolution = torch.tensor(res, dtype=torch.long, device=points_flat.device)
    network_strides = torch.tensor([res[2] * res[1], res[2], 1], dtype=torch.long, device=points_flat.device) # assumes row major ordering
    global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(points_flat.device)
    global_domain_size = global_domain_max - global_domain_min
    voxel_size = global_domain_size / fixed_resolution

    point_indices_3d = ((points_flat - global_domain_min) / voxel_size).to(network_strides)
    point_indices = (point_indices_3d * network_strides).sum(dim=1)

    del point_indices_3d

    PerfMonitor.add('point indices', ['point_indices'])

    # Filtering points outside global domain
    active_samples_mask = distance_mask

    if active_samples_mask is None:
        epsilon = 0.001
        active_samples_mask = torch.logical_and((points_flat > global_domain_min + epsilon).all(dim=1),
                                                (points_flat < global_domain_max - epsilon).all(dim=1))

        point_in_occupied_space = get_points_in_occupied_space(points_flat, occupancy_grid, cfg)

        active_samples_mask = torch.logical_and(active_samples_mask, point_in_occupied_space)
        del point_in_occupied_space
    #
    proper_index = torch.logical_and(point_indices >= 0,
                                     point_indices < num_networks)  # probably this is not needed if we check for points_flat <= global_domain_max
    active_samples_mask = torch.nonzero(torch.logical_and(active_samples_mask, proper_index), as_tuple=False).squeeze()
    del proper_index

    filtered_point_indices = point_indices[active_samples_mask]
    del point_indices

    PerfMonitor.add('filter', ['filter points'])

    # Sort according to network
    filtered_point_indices, reorder_indices = torch.sort(filtered_point_indices)

    PerfMonitor.add('sort', ['sort'])

    # make sure that also batch sizes are given for networks which are queried 0 points
    contained_nets, batch_size_per_network_incomplete = torch.unique_consecutive(filtered_point_indices,
                                                                                 return_counts=True)
    del filtered_point_indices
    batch_size_per_network = torch.zeros(num_networks, device=points_flat.device, dtype=torch.long)
    batch_size_per_network[contained_nets] = batch_size_per_network_incomplete
    batch_size_per_network = batch_size_per_network.cpu()

    # Reordering
    directions_flat = directions.unsqueeze(1).expand(points.size()).reshape(-1, 3)
    points_reordered = points_flat[active_samples_mask]
    directions_reordered = directions_flat[active_samples_mask]
    del points_flat, directions_flat
    # reorder so that points handled by the same network are packed together in the list of points
    points_reordered = points_reordered[reorder_indices]
    directions_reordered = directions_reordered[reorder_indices]
    PerfMonitor.add('reorder', ['reorder and backorder'])

    num_points_to_process = points_reordered.size(0) if points_reordered.ndim > 0 else 0
    # print("#points to process:", num_points_to_process, flush=True)
    if num_points_to_process == 0:
        return torch.zeros(num_rays, num_samples, 4, dtype=torch.float, device=points_reordered.device)

    # Convert global to local coordinates
    kilonerf_cuda.global_to_local(points_reordered, domain_mins, domain_maxs, batch_size_per_network, 1, 64)
    PerfMonitor.add('global to local', ['input transformation'])

    # Fourier features
    fourier_embedding_implementation = 'custom_kernel_v2'  # pytorch
    assert position_fourier_embedding is not None
    embedded_points = position_fourier_embedding(points_reordered.unsqueeze(0),
                                                 implementation=fourier_embedding_implementation).squeeze(0)

    del points_reordered
    assert direction_fourier_embedding is not None
    embedded_dirs = direction_fourier_embedding(directions_reordered.unsqueeze(0),
                                                implementation=fourier_embedding_implementation).squeeze(0)

    # print(embedded_points.shape, embedded_points.sum(), embedded_points.std())
    # print(embedded_dirs.shape, embedded_dirs.sum(), embedded_dirs.std())
    # print(batch_size_per_network)

    del directions_reordered
    embedded_points_and_dirs = [embedded_points, embedded_dirs]
    del embedded_points
    del embedded_dirs
    PerfMonitor.add('fourier features', ['input transformation'])

    # Network query
    raw_outputs = multi_network(embedded_points_and_dirs, batch_size_per_network, None)

    # print(raw_outputs.shape, raw_outputs.sum(), raw_outputs.std())

    # For debugging we can visualize which networks are responsible for which regions
    # This was also used to render the teaser figure.
    if has_flag(cfg, 'render_debug_network_color_map'):
        end_idx = 0
        batch_size_per_network_list = batch_size_per_network.tolist()
        for network_index in range(multi_network.num_networks):
            # res = iii
            ind = [(network_index // (res[2] * res[1])), (network_index // res[2]) % res[1], network_index % res[2]]
            start_idx = end_idx
            end_idx += batch_size_per_network_list[network_index]
            use_color_map = True
            if 'network_color_map_min' in cfg:
                for a, b in zip(ind, cfg['network_color_map_min']):
                    use_color_map = use_color_map and ind >= cfg['network_color_map_min']
            if 'network_color_map_max' in cfg:
                for a, b in zip(ind, cfg['network_color_map_max']):
                    use_color_map = use_color_map and ind <= cfg['network_color_map_max']
            if start_idx != end_idx and use_color_map:
                # assign random color to each network
                raw_outputs[start_idx:end_idx, :3] = debug_network_color_map[network_index]

    # Naive reordering is extremly fast even without any explicit measures to gurantee coherence => DeRF authors were telling lies
    raw_outputs_backordered = torch.empty_like(raw_outputs)
    raw_outputs_backordered[reorder_indices] = raw_outputs
    # raw_outputs_backordered = kilonerf_cuda.scatter_int32_float4(reorder_indices, raw_outputs)
    del raw_outputs
    raw_outputs_full = torch.zeros(num_rays * num_samples, 4, dtype=torch.float, device=raw_outputs_backordered.device)
    raw_outputs_full[active_samples_mask] = raw_outputs_backordered
    PerfMonitor.add('backorder', ['reorder and backorder'])

    raw_outputs_full = raw_outputs_full.view(num_rays, num_samples, -1)

    return raw_outputs_full


def render_rays(
    cfg,
    networks,
    ray_batch,
    N_samples=None,
    white_bkgd=False,
    raw_noise_std=0.,
    occupancy_grid=None,
    background_color=None,
    c2w=None,
    **kwargs,
):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = near * (1.-t_vals) + far * t_vals
    z_vals = z_vals.expand([N_rays, N_samples])

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    points_flat = pts.view(-1, 3)

    PerfMonitor.add('Point batching', ['batching'])

    if cfg.get("use_multires", 0):
        squared_distances = torch.sum((points_flat - c2w[:, -1]) ** 2, dim=1)

        # ind_far = squared_distances >= 40
        # ind_middle = squared_distances < 40
        # ind_near = squared_distances < 30

        ind_far = squared_distances >= 20
        ind_middle = squared_distances < 20
        ind_near = squared_distances < 14
        del squared_distances

        if cfg.get("use_rerun", 0):
            cols = torch.zeros_like(points_flat, dtype=torch.uint8)

            cols[ind_far] = torch.tensor([38, 70, 83], dtype=torch.uint8)
            cols[ind_middle] = torch.tensor([233, 196, 106], dtype=torch.uint8)
            cols[ind_near] = torch.tensor([231, 111, 81], dtype=torch.uint8)

            rr.log_points("world/points", points_flat.cpu().numpy(), colors=cols.cpu().numpy())

        raw_far = query_multi_network(networks, pts, viewdirs, occupancy_grid, cfg, distance_mask=ind_far, resolution=4, freq_index=0)
        raw_middle = query_multi_network(networks, pts, viewdirs, occupancy_grid, cfg, distance_mask=ind_middle, resolution=8, freq_index=0)
        raw_near = query_multi_network(networks, pts, viewdirs, occupancy_grid, cfg, distance_mask=ind_near, resolution=16, freq_index=0)

        raw = raw_far + raw_middle + raw_near
    elif cfg.get("use_multifreq", 1):
        squared_distances = torch.sum((points_flat - c2w[:, -1]) ** 2, dim=1)

        ind_far = squared_distances >= 16
        ind_near = ~ind_far

        PerfMonitor.add('distance mask', ['distance'])

        # global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(points_flat.device)
        # epsilon = 0.001
        # active_samples_mask = torch.logical_and((points_flat > global_domain_min + epsilon).all(dim=1),
        #                                         (points_flat < global_domain_max - epsilon).all(dim=1))
        #
        # point_in_occupied_space = get_points_in_occupied_space(points_flat, occupancy_grid, cfg)
        # active_samples_mask = torch.logical_and(active_samples_mask, point_in_occupied_space)

        active_samples_mask = get_points_in_occupied_space(points_flat, occupancy_grid, cfg)

        active_samples_mask_far = torch.logical_and(active_samples_mask, ind_far)
        active_samples_mask_near = torch.logical_and(active_samples_mask, ind_near)

        PerfMonitor.add('filter points', ['filter points'])

        del squared_distances

        if cfg.get("use_rerun", 0):
            cols = torch.zeros_like(points_flat, dtype=torch.uint8)

            cols[ind_far] = torch.tensor([38, 70, 83], dtype=torch.uint8)
            cols[ind_near] = torch.tensor([231, 111, 81], dtype=torch.uint8)

            rr.log_points("world/points", points_flat.cpu().numpy(), colors=cols.cpu().numpy())

        raw_far = query_multi_network(networks, pts, viewdirs, occupancy_grid, cfg, distance_mask=active_samples_mask_far, resolution=16, freq_index=1)
        raw_near = query_multi_network(networks, pts, viewdirs, occupancy_grid, cfg, distance_mask=active_samples_mask_near, resolution=16, freq_index=0)

        raw = raw_far + raw_near
    else:
        raw = query_multi_network(networks, pts, viewdirs, occupancy_grid, cfg, distance_mask=None, resolution=16, freq_index=0)

    no_color_sigmoid = has_flag(cfg, 'no_color_sigmoid')
    rgb_map, disp_map, acc_map, weights, depth_map, alpha, transmittance = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, background_color,
        pytest=False, no_color_sigmoid=no_color_sigmoid)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}

    # for k in ret:
    #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and False:
    #         print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def batchify_rays(cfg, networks, rays_flat, occupancy_grid, chunk=1024*32, c2w=None, background_color=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """

    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(cfg, networks, rays_flat[i:i+chunk], background_color=background_color, occupancy_grid=occupancy_grid, c2w=c2w, **kwargs)

        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

    return all_ret


def render(cfg, networks, intrinsics, occupancy_grid, chunk=1024*32, c2w=None, near=0., far=1., background_color=None, **kwargs):
    PerfMonitor.add('start')
    PerfMonitor.is_active = has_flag(cfg, 'performance_monitoring')

    # rays_o, rays_d = get_rays(intrinsics, c2w)

    rays_o2, rays_d2 = get_rays_np(intrinsics, c2w.cpu().numpy())
    rays_o = torch.from_numpy(np.copy(rays_o2)).to(device)
    rays_d = torch.from_numpy(np.copy(rays_d2)).to(device)

    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    PerfMonitor.add('ray directions', ['preprocessing'])

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    log_points = []

    # for i, (o, d) in enumerate(zip(rays_o, rays_d)):
    #     o = o.cpu().numpy()
    #     d = d.cpu().numpy() * far
    #
    #     log_points.append(o)
    #     log_points.append(d)

        # rr.log_arrow(f"world/rays/{i:06d}", origin=o, vector=d)
        #
        # print(o, d)

    # rr.log_line_segments("world/rays", log_points)

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(cfg, networks, rays, occupancy_grid, chunk, background_color=background_color, c2w=c2w, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    PerfMonitor.is_active = True
    PerfMonitor.add('integration', ['integration'])
    elapsed_time = PerfMonitor.log_and_reset(has_flag(cfg, 'performance_monitoring'))

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    all_ret = ret_list + [elapsed_time] + [ret_dict]

    return all_ret


def render_path(networks, render_poses, intrinsics, chunk, occupancy_grid, render_kwargs, savedir, cfg, images, render_factor=0):
    intrinsics = copy(intrinsics)
    if render_factor != 1 and render_factor != 0:
        intrinsics.H = int(intrinsics.H / render_factor)
        intrinsics.W = int(intrinsics.W / render_factor)
        intrinsics.fx = intrinsics.fx / render_factor
        intrinsics.fy = intrinsics.fy / render_factor
        intrinsics.cx = intrinsics.cx / render_factor
        intrinsics.cy = intrinsics.cy / render_factor

    if cfg.get("use_rerun", 0):
        width = intrinsics.W
        height = intrinsics.H

        rr.log_pinhole("world/camera/image", child_from_parent=[[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]], width=width, height=height)

    c2ws = [x[:3, :4] for x in render_poses]

    global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(device)
    global_domain_min = global_domain_min.cpu().numpy()
    global_domain_max = global_domain_max.cpu().numpy()
    domain_size_half = (global_domain_max - global_domain_min) / 2
    center = global_domain_max + global_domain_min

    print(global_domain_min, global_domain_max)

    if cfg.get("use_rerun", 0):
        rr.log_obb("world/global_domain", half_size=domain_size_half, position=center)

    psnr_list = []
    times_list = []

    for i, c2w in enumerate(tqdm(c2ws)):
        Logger.write(f'Rendering image {i}')

        if cfg.get("use_rerun", 0):
            c2w_num = c2w.cpu().numpy()

            c2w_num = c2w_num @ LOGGING_TRANSFORM

            t = rr.TranslationAndMat3(translation=c2w_num[:, -1], matrix=c2w_num[:, :3])

            rr.log_transform3d("world/camera", t)

        rgb, disp, acc, elapsed_time = render(cfg, networks, intrinsics, occupancy_grid, chunk=chunk, c2w=c2w, **render_kwargs)[:4]

        rgb8 = to8b(rgb.cpu().numpy())
        out_file = savedir / f"{i:03d}.png"
        # out_file2 = savedir / f"c{i:03d}.png"

        if cfg.get("use_rerun", 0):
            rr.log_image("world/camera/image/rgb", rgb8)

        Logger.write(f"Writing image file to {out_file}")

        imageio.imwrite(str(out_file), rgb8)
        # imageio.imwrite(str(out_file2), to8b(images[i]))

        if cfg.get("calc_metrics", False) and (render_factor == 1 or render_factor == 0):
            gt_img_pytorch = torch.tensor(images[i], device=device)
            mse = img2mse(rgb, gt_img_pytorch)
            psnr = mse2psnr(mse)

            psnr_list.append(psnr.item())
            times_list.append(elapsed_time)

            print(i,  mse.item(), psnr.item())

        # break
        if i > 10:
            break
    if psnr_list:
        print("Average psnr:", sum(psnr_list) / len(psnr_list))
        print("Average time:", sum(times_list) / len(times_list))


def fill_cfg(render_cfg_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    # "Render" config overwrites the config
    if render_cfg_path is not None:
        render_cfg = load_yaml_as_dict(render_cfg_path)
        for key in render_cfg:
            cfg[key] = render_cfg[key]

    if 'rng_seed' in cfg:
        np.random.seed(cfg['rng_seed'])
        torch.manual_seed(cfg['rng_seed'])

    # Copy config values from distillation phases to top level
    def copy_to_top_level(cfg):
        if 'final' in cfg:
            for key in cfg['final']:
                cfg[key] = cfg['final'][key]
        elif 'discovery' in cfg:
            for key in cfg['discovery']:
                cfg[key] = cfg['discovery'][key]

    finetuning_distilled = 'distilled_cfg_path' in cfg
    if finetuning_distilled:
        distilled_cfg = load_yaml_as_dict(cfg['distilled_cfg_path'])
        copy_to_top_level(distilled_cfg)

        # Add configs in distilled config to this config
        for key in cfg:
            distilled_cfg[key] = cfg[key]
        cfg = distilled_cfg
    else:
        copy_to_top_level(cfg)

    if has_flag(cfg, 'visualize_global_domain') and 'occupancy_cfg_path' in cfg:
        del cfg['occupancy_cfg_path']

    ConfigManager.init(cfg)

    return cfg


def load_networks(finetune_dir: Path, cfg: Dict[str, Any]):
    global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(torch.device('cpu'))
    global_domain_size = global_domain_max - global_domain_min

    cp_multi = {}

    freq_pairs = [(cfg['num_frequencies'], cfg['num_frequencies_direction']), (4, 2)]

    for freq_i, (num_freq, num_freq_dir) in enumerate(freq_pairs):
        position_num_input_channels, position_fourier_embedding = create_multi_network_fourier_embedding(1, num_freq)
        direction_num_input_channels, direction_fourier_embedding = create_multi_network_fourier_embedding(1, num_freq_dir)

        cp_multi[freq_i] = {}

        if freq_i == 1:
            finetune_dir = finetune_dir.with_name(finetune_dir.name + "_Low")

        for res_dir in finetune_dir.iterdir():
            if not res_dir.is_dir():
                continue

            checkpoint_filenames = [f for f in res_dir.iterdir() if 'checkpoint' in f.name]

            if not checkpoint_filenames:
                continue

            # checkpoint_file = sorted(checkpoint_filenames)[-1]
            checkpoint_file = res_dir / "checkpoint_0040000.pth"

            res = int(res_dir.name)

            if res not in {4, 8, 16}:
                continue

            Logger.write(f'Loading {checkpoint_file} for resolution {res}')

            network_resolution = torch.tensor([res, res, res], dtype=torch.long, device=torch.device('cpu'))
            num_networks = res * res * res

            Logger.write(f"Creating model with {num_networks} networks ({position_num_input_channels})")

            # TODO: check if multimatmul(_differentiable) needed here as these require kilonerf_cuda to be initialized
            model = create_multi_network(num_networks, position_num_input_channels, direction_num_input_channels, 4, 'multimatmul_differentiable', cfg).to(device)
            Logger.write(f"Done creating model with {num_networks} networks")

            network_voxel_size = global_domain_size / network_resolution

            domain_mins = []
            domain_maxs = []
            for coord in itertools.product(*[range(r) for r in [res, res, res]]):
                coord = torch.tensor(coord, device=torch.device('cpu'))
                domain_min = global_domain_min + network_voxel_size * coord
                domain_max = domain_min + network_voxel_size
                domain_mins.append(domain_min.tolist())
                domain_maxs.append(domain_max.tolist())
            domain_mins = torch.tensor(domain_mins, device=device)
            domain_maxs = torch.tensor(domain_maxs, device=device)

            debug_network_color_map = None
            if has_flag(cfg, 'render_debug_network_color_map'):
                debug_network_color_map = torch.tensor([1.0, 0.0, 0.0])

                if freq_i == 1:
                    debug_network_color_map = torch.tensor([0.0, 0.0, 1.0])

                # if res == 8:
                #     debug_network_color_map = torch.tensor([0.0, 1.0, 0.0])
                # elif res == 4:
                #     debug_network_color_map = torch.tensor([0.0, 0.0, 1.0])

                debug_network_color_map = debug_network_color_map.repeat(res * res * res, 1)
                # debug_network_color_map = torch.cat(channels, dim=-1)

                print(debug_network_color_map.shape)

            cp = torch.load(checkpoint_file)

            model.load_state_dict(cp['model_state_dict'])

            cp_multi[freq_i][res] = {
                "weights": cp,
                "model": model,
                "domain_mins": domain_mins,
                "domain_maxs": domain_maxs,
                "debug_network_color_map": debug_network_color_map,
                "position_fourier_embedding": position_fourier_embedding,
                "direction_fourier_embedding": direction_fourier_embedding,
            }

    return cp_multi


def load_occupancy(cfg: Dict[str, Any]):
    occupancy_cfg = load_yaml_as_dict(cfg['occupancy_cfg_path'])
    if 'occupancy' not in cfg:
        cfg['occupancy'] = {}
    for key in occupancy_cfg:
        cfg['occupancy'][key] = occupancy_cfg[key]
    Logger.write('Loading occupancy grid from {}'.format(cfg['occupancy_log_path']))
    occupancy_grid = torch.load(cfg['occupancy_log_path']).reshape(-1)

    global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max(device)
    global_domain_size = global_domain_max - global_domain_min

    res = cfg['occupancy']['resolution']
    occupancy_resolution = torch.tensor(res, dtype=torch.long, device=device)
    occupancy_voxel_size = global_domain_size / occupancy_resolution
    occupancy_voxel_half_size = occupancy_voxel_size / 2
    occupancy_voxel_centers = []

    for dim in range(3):
        occupancy_voxel_centers.append(torch.linspace(global_domain_min[dim] + occupancy_voxel_half_size[dim],
                                                      global_domain_max[dim] - occupancy_voxel_half_size[dim],
                                                      res[dim]))
    occupancy_voxel_centers = torch.stack(torch.meshgrid(*occupancy_voxel_centers), dim=3).view(-1, 3)

    return occupancy_grid, occupancy_voxel_centers


def render_main(log_path: str, render_cfg_path: str, cfg: Dict[str, Any]):
    # Required for fast training
    kilonerf_cuda.init_stream_pool(16)
    kilonerf_cuda.init_magma()

    cfg = fill_cfg(render_cfg_path, cfg)

    if cfg.get("use_rerun", 0):
        rr.init("kilonerf", spawn=True)

    log_path = Path(log_path)

    Logger.write('Using GPU: {}'.format(torch.cuda.get_device_name(0)))

    test_traj_path = cfg['test_traj_path'] if 'test_traj_path' in cfg else None
    images, poses, intrinsics, near, far, background_color, render_poses, i_split = load_nsvf_dataset(
        cfg['dataset_dir'], cfg['testskip'], test_traj_path)

    i_train, i_val, i_test, i_test2 = i_split
    if i_test.size == 0:
        i_test = i_val

    print(i_train.shape, i_val.shape, i_test.shape)
    # images = images[..., :3]
    print('Converting alpha to white.')
    images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])

    background_color = torch.ones(3, dtype=torch.float, device=device)
    # global_domain_min, global_domain_max = ConfigManager.get_global_domain_min_and_max()

    near = cfg.get("near", near)
    far = cfg.get("far", far)

    render_subset = cfg.get("render_subset", "custom_path")
    i_render = {"train": i_train, "val": i_val, "test": i_test, "render": i_test2}[render_subset]

    render_poses = np.array(poses[i_render])
    images = images[i_render]

    render_kwargs_train = {
        'N_samples': cfg['num_samples_per_ray'],
        'white_bkgd': cfg['blender_white_background'],
        'raw_noise_std': cfg['raw_noise_std'],
        'near': near,
        'far': far,
        'background_color': background_color,
    }

    networks = load_networks(log_path, cfg)

    print("Got networks", networks.keys())

    occupancy_grid, occupancy_voxel_centers = load_occupancy(cfg)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    print("Start rendering")

    for network_inner in networks.values():
        for n in network_inner.values():
            n["model"].eval()


    testsavedir = log_path / f"render_{cfg.get('render_subset', 'unknown')}"
    if render_cfg_path is not None:
        render_cfg_name = Path(render_cfg_path).stem

        if cfg.get("render_debug_network_color_map", False):
            render_cfg_name += "_networks"

        testsavedir = testsavedir.with_name(render_cfg_name)
    testsavedir.mkdir(exist_ok=True, parents=True)

    print('test poses shape', render_poses.shape)

    render_kwargs_test = render_kwargs_train.copy()

    render_path(
        networks,
        render_poses,
        intrinsics,
        cfg['chunk_size'],
        occupancy_grid,
        render_kwargs_test,
        testsavedir,
        cfg=cfg,
        images=images,
        render_factor=cfg['render_factor'],
    )


def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # sneaky
    cfg, log_path, render_cfg_path = parse_args_and_init_logger('default.yaml', parse_render_cfg_path=True)

    with torch.no_grad():
        render_main(log_path, render_cfg_path, cfg)


if __name__ == '__main__':
    main()
