from src.train import train_mnist

from argparse import ArgumentParser, Namespace

def run(args: Namespace):
    train_mnist(args)


if __name__ == "__main__":
    args = Namespace(
        epochs=20
    )
    run(args)