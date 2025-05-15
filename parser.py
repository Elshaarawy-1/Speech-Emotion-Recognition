import argparse

def default_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100, help="Specify number of epochs")

    parser.add_argument(
        "-B",
        "--batch_size",
        default=32,
        type=int,
        help="Default batch size is 32. Reduce this if the data doesn't fit in your GPU.",
    )

    parser.add_argument(
        "-LR", "--learning_rate", type=float, default=1e-5, help="Default Learning rate is 1e-5."
    )

    parser.add_argument(
        "--model_type",
        default=1,
        help='Specifies the specific architecture to be used. Check README for more info. Defaults to "Ravdess".',
    )
    return parser.parse_args()