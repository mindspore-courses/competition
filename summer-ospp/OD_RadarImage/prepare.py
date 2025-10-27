import argparse

from datasets import prepare
from utils.config import load_config
from utils.misc import set_seed

def main(src: str,cfg: str,dst: str):
    """ Data preparation for subsequent model training or evaluation.

    Arguments:
        scr: Source directory path to the raw dataset folder.
        cfg: Path to the configuration file.
        dst: Destination directory to save the processed dataset files.
    """
    # Load dataset configuration
    config = load_config(cfg)

    # Set global random seed
    set_seed(config['computing']['seed'])

    # Prepare dataset
    preperator = prepare(config['dataset'], config)
    preperator.prepare(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DPRT data preprocessing')
    parser.add_argument('--src', type=str, default='./mini_data',
                        help="Path to the raw dataset folder.")
    parser.add_argument('--cfg', type=str, default='./config/kradar.json',
                        help="Path to the configuration file.")
    parser.add_argument('--dst', type=str, default='./processed_data',
                        help="Path to save the processed dataset.")
    args = parser.parse_args()

    main(src=args.src, cfg=args.cfg, dst=args.dst)