#######################################
#     DO NOT CHANGE THESE IMPORTS

import sys
sys.path.insert(0, "avalanche")

#######################################

import argparse
import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import numpy as np

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)

from benchmarks import get_cifar_based_benchmark
from utils.competition_plugins import (
    GPUMemoryChecker,
    RAMChecker,
    TimeChecker
)

# Added extra imports -- not originally in CLVISION challenge 2023 DevKit
import os
from utils.facil_logger import FACIL_Logger, FileOutputDuplicator

# Extended ResNet18 with our Horde functionalities
from models import HordeSlimResNet18

# Our proposed strategy: Horde
from strategies.horde import HordeMLStrat


def main(args):
    # --- Device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")
    # --- Benchmark
    benchmark = get_cifar_based_benchmark(scenario_config=args.config_file, seed=args.seed)
    # --- Model
    model = HordeSlimResNet18(n_classes=benchmark.n_classes)
    try:
        # --- Logger and metrics
        results_path = os.path.join("results", f"{args.run_name}_{os.path.splitext(args.config_file)[0]}")
        os.makedirs(results_path, exist_ok=True)
        # Duplicate standard outputs
        sys.stdout = FileOutputDuplicator(sys.stdout, os.path.join(results_path, 'stdout.txt'), 'w')
        sys.stderr = FileOutputDuplicator(sys.stderr, os.path.join(results_path, 'stderr.txt'), 'w')
        # Custom logger for compact output and JSON support
        # interactive_logger = InteractiveLogger()  # original logger
        facil_logger = FACIL_Logger(file=sys.stdout)
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=False, epoch=True, experience=False, stream=True),
            loss_metrics(minibatch=False, epoch=True, experience=False, stream=True),
            forgetting_metrics(experience=True),
            loggers=[facil_logger],  # loggers=[interactive_logger],
        )

        # --- Competition Plugins
        # DO NOT REMOVE OR CHANGE THESE PLUGINS:
        competition_plugins = [
            TimeChecker(max_allowed=500)
        ]
        # --- Your Plugins -- we do not need plugins, everything is handled in `strategies/horde.py`
        plugins = []
        # --- Strategy
        cl_strategy = HordeMLStrat(model, torch.optim.Adam(model.parameters(), lr=0.001),
                                   # General arguments
                                   criterion=torch.nn.CrossEntropyLoss(),
                                   train_mb_size=32,
                                   train_epochs=args.num_epochs,
                                   eval_mb_size=512,
                                   device=device, plugins=competition_plugins + plugins, evaluator=eval_plugin,
                                   # Horde arguments
                                   number_feature_extractors=args.num_fes,
                                   iterations_for_mean=3,
                                   ml_margin=1.0,
                                   alpha=args.alpha,
                                   acc_thr=1.0,
                                   num_ml_dims=32,
                                   best_loss_model=False,
                                   use_curr_cls_ph2=True,
                                   num_sim_feats=1,
                                   unk_mean=args.unk_mean,
                                   unk_std=1.0,
                                   )

        # --- Training Loops
        for experience in benchmark.train_stream:
            # Removed the evaluation for the train stream
            cl_strategy.train(experience, num_workers=args.num_workers, eval_streams=[])
            # Save the model after each experience
            torch.save(cl_strategy.model, os.path.join(results_path,
                                                       f"model-exp_{cl_strategy.clock.train_exp_counter}.pth"))

        # --- Make prediction on test-set samples
        predictions = predict_test_set(cl_strategy.model, benchmark.test_stream[0].dataset, device)

        # Save predictions
        output_name = f"pred_{args.config_file.split('.')[0]}_{args.run_name}.npy"
        np.save(output_name, predictions)

    except Exception as e:
        sys.stdout.flush()
        sys.stderr.flush()
        raise e


def predict_test_set(model, test_set, device):
    print("Making prediction on test-set samples")
    model.eval()
    dataloader = DataLoader(test_set, batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for (x, _, _) in dataloader:
            pred = model(x.to(device)).detach().cpu()
            preds.append(pred)
    preds = torch.cat(preds, dim=0)
    preds = torch.argmax(preds, dim=1).numpy()
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    parser.add_argument("--config_file", type=str, default="config_s1.pkl")
    parser.add_argument("--run_name", type=str, default="horde-s1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    # Horde arguments
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_fes", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=-1.0, help="Balance (1-alpha)*CE-loss and alpha*ML-loss,"
                                                                  "or use an adaptive strategy with -1.0.")
    parser.add_argument("--unk_mean", default="feats", choices=["zeros", "feats", "noise"])

    args = parser.parse_args()
    main(args)
