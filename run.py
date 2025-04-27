# run.py
import argparse
from utils.train import EnRun
from utils.config import EnConfig
from distutils.util import strtobool

def main(args):
    EnRun(EnConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        dataset_name=args.dataset,
        num_hidden_layers=args.num_hidden_layers,
        use_context=args.use_context,
        use_attnFusion=args.use_attnFusion,
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-6, help='learning rate, recommended: 5e-6 for mosi, mosei, 1e-5 for sims')
    parser.add_argument('--dataset', type=str, default='mosi', help='dataset name: mosi, mosei, sims')
    parser.add_argument('--num_hidden_layers', type=int, default=5, help='number of hidden layers for cross-modality encoder')
    parser.add_argument('--use_context', type=lambda x: bool(strtobool(x)), default=True, help='enable context reasoning')
    parser.add_argument('--use_attnFusion', type=lambda x: bool(strtobool(x)), default=True, help='enable attention fusion')
    args = parser.parse_args()
    main(args)

