import torch
from pathlib import Path
from utils import parse_args_and_init_logger, Logger
from run_nerf import train

def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor') # sneaky
    cfg, log_path, render_cfg_path = parse_args_and_init_logger('default.yaml', parse_render_cfg_path=True)

    print(log_path)
    print(render_cfg_path)

    print(cfg)

    base_path = Path(log_path)

    distill_base_path = Path(cfg["distilled_checkpoint_path"]).parent

    for res in (16, 8, 4):
    # for res in (8, 4, 2, 1):
    # for res in (4, ):
    # for res in (4, ):
        log_path = base_path / str(res)
        log_path.mkdir(exist_ok=True, parents=True)

        distill_path = distill_base_path / str(res) / "checkpoint.pth"
        distill_path.parent.mkdir(parents=True, exist_ok=True)

        Logger.write(f"Training with {distill_path}")

        cfg["distilled_checkpoint_path"] = distill_path

        train(cfg, str(log_path), str(render_cfg_path) if render_cfg_path else None, resolution_overwrite=[res, res, res])
    exit(0)


if __name__ == '__main__':
    main()
