from avalanche.training import GEM
import torch
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST, RotatedMNIST, SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training import AGEM

import random
import numpy as np

def main(args):
    # Set seeds
    seed = 123  # or any other number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Device config
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    # model
    model = SimpleMLP(num_classes=10)

    # Here we show all the MNIST variation we offer in the "classic" benchmarks
    if args.mnist_type == "permuted":
        benchmark = PermutedMNIST(n_experiences=5, seed=1)
    elif args.mnist_type == "rotated":
        benchmark = RotatedMNIST(
            n_experiences=5, rotations_list=[30, 60, 90, 120, 150], seed=1
        )
    else:
        benchmark = SplitMNIST(n_experiences=5, seed=1)

    # Than we can extract the parallel train and test streams
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = CrossEntropyLoss()

    # Continual learning strategy with default logger
    cl_strategy = AGEM(
        model,
        optimizer,
        criterion,
        train_mb_size=32,
        train_epochs=2,
        eval_mb_size=32,
        device=device,
        patterns_per_exp=256,     # ✅ required argument
    )

    # train and test loop
    results = []
    for train_task in train_stream:
        print("Current Classes: ", train_task.classes_in_this_experience)
        cl_strategy.train(train_task)
        results.append(cl_strategy.eval(test_stream))
    import pandas as pd
    df = pd.DataFrame(results)
    # df.to_csv(".\\examples\\GEM\\gem_example_results\\gem_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnist_type",
        type=str,
        default="split",
        choices=["rotated", "permuted", "split"],
        help="Choose between MNIST variations: " "rotated, permuted or split.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()

    main(args)

