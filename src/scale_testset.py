import sys
from pathlib import Path


def main(dataset_dir: Path):
    pose_path = dataset_dir / "pose"
    rgb_path = dataset_dir / "rgb"

    for infile in pose_path.glob("2_*.txt"):
        lines = infile.read_text().splitlines()

        numbers = [
            [float(n) for n in line.split()] for line in lines
        ]

        for i in range(3):
            numbers[i][3] *= 1.5

        out_text = "\n".join(" ".join(str(n) for n in row) for row in numbers)

        out_file = infile.with_name(f"3{infile.name[1:]}")
        out_file.write_text(out_text)

        out_rgb = rgb_path / (out_file.stem + ".png")
        out_rgb.touch(exist_ok=True)

        print(out_file, out_rgb)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
