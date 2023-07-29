import sys
from pathlib import Path
import shutil


def main(dataset_dir: Path):
    pose_path = dataset_dir / "pose"
    rgb_path = dataset_dir / "rgb"
    bbox_file = dataset_dir / "bbox.txt"

    bounds = [float(f) for f in bbox_file.read_text().split()][:-1]
    abs_bounds = [abs(f) for f in bounds]

    print(bounds)

    total_max = max(abs_bounds)

    total_size = total_max * 2

    new_dataset_dir = dataset_dir.with_name(dataset_dir.name + "_norm")
    new_dataset_dir.mkdir(exist_ok=True, parents=True)

    bbox_file_out = new_dataset_dir / "bbox.txt"
    bbox_file_out.write_text(f"-0.5 -0.5 -0.5 0.5 0.5 0.5 1.0")

    new_rgb_folder = new_dataset_dir / 'rgb'
    new_pose_folder = new_dataset_dir / 'pose'

    # new_rgb_folder.mkdir(exist_ok=True, parents=True)
    new_pose_folder.mkdir(exist_ok=True, parents=True)

    shutil.copy(dataset_dir / "intrinsics.txt", new_dataset_dir)
    shutil.copytree(rgb_path, new_rgb_folder)

    for pose_file in pose_path.iterdir():
        m = [
            [float(f) for f in line.split()] for line in pose_file.read_text().splitlines()
        ]

        for i in range(3):
            m[i][3] /= total_size

        out_text = '\n'.join(
            " ".join(str(i) for i in line) for line in m
        )

        outfile = new_pose_folder / pose_file.name
        outfile.write_text(out_text)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
