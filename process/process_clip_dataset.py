import argparse
import json
from pathlib import Path


def main(args):
    num_parts = args.num_parts
    input_dir = args.input_dir
    output_dir = args.output_dir
    

    all_clips = [p for p in Path(input_dir).glob("*") if p.is_dir()]
    num_clips = len(all_clips)

    # 将 all_clips 均分为 num_parts 部分
    k, m = divmod(num_clips, num_parts)
    clip_parts = [
        all_clips[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(num_parts)
    ]
    clip_parts_str = [[str(p) for p in part] for part in clip_parts]

    with open(Path(output_dir) / "clip_parts.json", "w") as f:
        json.dump(clip_parts_str, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-parts", type=int, default=4)
    args = parser.parse_args()
    main(args)


