import torch
from pathlib import Path
from utils import parse_args_and_init_logger, Logger
from run_nerf import train


def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor') # sneaky
    cfg, log_path, render_cfg_path = parse_args_and_init_logger('default.yaml', parse_render_cfg_path=True)

    base_path = Path(log_path)

    distill_base_path = Path(cfg["distilled_checkpoint_path"]).parent.parent

    res = 16

    log_path = base_path / str(res)
    distill_path = distill_base_path / str(res) / "checkpoint.pth"

    Logger.write(f"Training with {distill_path}")

    cfg["distilled_checkpoint_path"] = distill_path

    train(cfg, str(log_path), str(render_cfg_path), resolution_overwrite=[res, res, res])
    exit(0)


if __name__ == '__main__':
    main()
