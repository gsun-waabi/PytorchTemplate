import argparse
from core.runner import Runner

def lets_go(args):
    mode = args.mode
    runner = Runner(args)
    if mode == "train":
        runner.train_model()
    else:
        print(f"ERROR: Mode {mode} not supported!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Runner Mode', dest='mode')

    train_parser = subparsers.add_parser("train")

    ### Sky Encoder Parser Args
    train_parser.add_argument("--expr_root", type=str, default="expr")
    train_parser.add_argument("--trial_name", type=str, required=True)
    train_parser.add_argument("--train_root", type=str, default="data/exrs/train")
    train_parser.add_argument("--val_root", type=str, default="data/exrs/val")
    train_parser.add_argument("--val_limit", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=42)

    train_parser.add_argument("--batch_size", type=int, default=8)

    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--lr_decay_rate", type=int, default=1000)
    train_parser.add_argument("--lr_decay_factor", type=float, default=0.3)

    train_parser.add_argument("--epochs", type=int, default=4000)
    train_parser.add_argument("--resume_epoch", type=int, default=0)
    train_parser.add_argument("--val_rate", type=int, default=100)
    train_parser.add_argument("--save_rate", type=int, default=500)
    train_parser.add_argument("--log_rate", type=int, default=10)
            
    args = parser.parse_args()
    lets_go(args)

