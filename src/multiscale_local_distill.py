from local_distill import train, parse_args_and_init_logger
from pathlib import Path


def main():
    cfg, log_path = parse_args_and_init_logger()

    name = Path(log_path).name

    # for res in (16, 8, 4, 2, 1):
    for res in (16, 8, 4):
        print("Training with resolution", res)
        cfg["fixed_resolution"] = (res, res, res)

        log_path = Path(__file__).parent / "logs" / "paper" / "distill_multi" / name / str(res)
        log_path.mkdir(exist_ok=True, parents=True)

        print(log_path)

        train(cfg, str(log_path))

    print(cfg)

    # restarting_job = train(cfg, log_path)
    # exit(3 if restarting_job else 0)
    
	
if __name__ == '__main__':
	main()
	