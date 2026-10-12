import argparse
from pathlib import Path

from utils import create_input_files

if __name__ == '__main__':
    workspace = Path(__file__).resolve().parent.parent
    default_raw = workspace / 'data' / 'raw'
    default_output = workspace / 'data' / 'processed'

    parser = argparse.ArgumentParser(description='Prepare COCO captioning input files.')
    parser.add_argument('--json-path', type=Path, default=default_raw / 'dataset_coco.json')
    parser.add_argument('--image-folder', type=Path, default=default_raw)
    parser.add_argument('--output-folder', type=Path, default=default_output)
    args = parser.parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)

    create_input_files(dataset='coco',
                       json_path=str(args.json_path),
                       image_folder=str(args.image_folder),
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=str(args.output_folder),
                       max_len=50)
