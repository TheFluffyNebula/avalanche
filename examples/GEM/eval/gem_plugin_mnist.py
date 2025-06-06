from avalanche.training import GEM
import torch
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import PermutedMNIST, RotatedMNIST, SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training import GEM
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger

def main(args):
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

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=benchmark.n_classes, save_image=False, stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger()],
        strict_checks=False
    )

    # Continual learning strategy with default logger
    cl_strategy = GEM(
        model,
        optimizer,
        criterion,
        train_mb_size=32,
        train_epochs=2,
        eval_mb_size=32,
        device=device,
        patterns_per_exp=256,     # ✅ required argument
        memory_strength=0.5,       # optional, but good default from the paper
        evaluator=eval_plugin,
    )

    # train and test loop
    results = []
    for train_task in train_stream:
        print("Current Classes: ", train_task.classes_in_this_experience)
        cl_strategy.train(train_task)
        results.append(cl_strategy.eval(test_stream))
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(".\\examples\\GEM\\eval\\results\\gem_eval.csv")

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

