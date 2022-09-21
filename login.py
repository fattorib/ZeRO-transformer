import wandb
import argparse

def parse():
    parser = argparse.ArgumentParser(description="Transformer Training")

    parser.add_argument(
        "--key",
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    wandb.login(key = args.key)