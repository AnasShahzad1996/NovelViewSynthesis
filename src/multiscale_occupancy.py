import torch
import sys
import yaml
from pathlib import Path
import numpy as np

from skimage.measure import block_reduce


def pad_array(array, target_size):
    current_size = array.shape
    pad_width = tuple((0, max(target - current, 0)) for target, current in zip(target_size, current_size))
    padded_array = np.pad(array, pad_width, mode='constant')
    return padded_array


def viz(config_file: Path, occ_file: Path, out_path: Path):
    with config_file.open() as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    occupancy_grid = torch.load(occ_file).reshape(-1).cpu()

    occupancy_grid = occupancy_grid.view(cfg['resolution'] + [-1]).squeeze()

    print('{} out of {} voxels are occupied. {:.2f}%'.format(occupancy_grid.sum().item(), occupancy_grid.numel(),
                                                       100 * occupancy_grid.sum().item() / occupancy_grid.numel()))

    print(cfg['resolution'] + [-1], occupancy_grid.shape)

    occ_padded = pad_array(occupancy_grid.numpy(), (256, 256, 256))

    print(occ_padded.shape)

    out_path.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    for bs in (1, 2, 4, 8, 16, 32, 64, 128):
        downscaled_grid = block_reduce(occ_padded, block_size=(bs, bs, bs), func=np.any)

        t = torch.from_numpy(downscaled_grid).to(dev)
        filename = out_path / f"{t.shape[0]}.pth"
        torch.save(t, filename)
        print(filename, downscaled_grid.shape)


if __name__ == '__main__':
    name = sys.argv[1]
    config_file = Path(__file__).parent / "cfgs" / "paper" / "pretrain_occupancy" / f"{name}.yaml"
    occ_file = Path(__file__).parent / "logs" / "paper" / "pretrain_occupancy" / name / "occupancy.pth"
    out_path = Path(__file__).parent / "logs" / "paper" / "pretrain_occupancy_multi" / name

    viz(config_file, occ_file, out_path)