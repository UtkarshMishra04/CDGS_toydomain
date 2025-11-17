import argparse
import torch
import numpy as np
from data import MultiModalDataset, UniformDataset
from models import SimpleDiffusionModel
from samplers import CDGS
import time
import os
import json

# Configure device between CUDA, MPS, and CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


def build_multimodal_datasets(num_samples: int, seed: int):
    """Build multimodal (Gaussian) datasets."""
    start_dataset = MultiModalDataset(
        num_samples=num_samples,
        start_means=[0.0],
        start_stds=[0.2],
        end_means=[0.75, -0.75],
        end_stds=[0.1, 0.1],
        transition_matrix=np.array([[1, 1]], dtype=bool),
        seed=seed,
    )

    bridge_dataset = MultiModalDataset(
        num_samples=num_samples,
        start_means=[0.75, -0.75],
        start_stds=[0.1, 0.1],
        end_means=[0.75, -0.75],
        end_stds=[0.1, 0.1],
        transition_matrix=np.array([[1, 0], [0, 1]], dtype=bool),
        seed=seed,
    )

    end_dataset = MultiModalDataset(
        num_samples=num_samples,
        start_means=[0.75, -0.75],
        start_stds=[0.1, 0.1],
        end_means=[0.0],
        end_stds=[0.2],
        transition_matrix=np.array([[1], [1]], dtype=bool),
        seed=seed,
    )

    return start_dataset, bridge_dataset, end_dataset


def build_uniform_datasets(num_samples: int, seed: int):
    """Build uniform datasets."""
    start_dataset = UniformDataset(
        num_samples=num_samples,
        start_ranges=[(-0.25, 0.25)],  # X_mid
        end_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        transition_matrix=np.array([[1, 1]], dtype=bool),
        seed=seed,
    )

    bridge_dataset = UniformDataset(
        num_samples=num_samples,
        start_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        end_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        transition_matrix=np.array([[1, 0], [0, 1]], dtype=bool),
        seed=seed,
    )

    end_dataset = UniformDataset(
        num_samples=num_samples,
        start_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        end_ranges=[(-0.25, 0.25)],  # X_mid
        transition_matrix=np.array([[1], [1]], dtype=bool),
        seed=seed,
    )

    return start_dataset, bridge_dataset, end_dataset


def main(args):
    # Build datasets based on type
    num_samples = 1000
    seed = 42

    if args.dataset == "multimodal":
        start_dataset, bridge_dataset, end_dataset = build_multimodal_datasets(num_samples, seed)
    else:  # uniform
        start_dataset, bridge_dataset, end_dataset = build_uniform_datasets(num_samples, seed)

    # Build model paths based on dataset type
    diffusion_model_paths = {
        "start": f"./checkpoints/{args.dataset}_start_model.pth",
        "bridge": f"./checkpoints/{args.dataset}_bridge_model.pth",
        "end": f"./checkpoints/{args.dataset}_end_model.pth",
    }

    num_bridges = args.horizon_length - 2
    sampler = CDGS(
        model_paths=diffusion_model_paths,
        device=str(device),
        model_type=SimpleDiffusionModel,
        num_bridges=num_bridges,
        num_resampling_steps=args.num_resampling_steps,
        pruning_start=args.pruning_start,
        pruning_end=args.pruning_end,
        pruning_top_K=args.pruning_top_K,
        enable_pruning=args.enable_pruning,
    )

    start = time.monotonic()
    samples = sampler.sample(
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
    )
    samples = samples[: args.num_samples_to_generate]
    time_taken = time.monotonic() - start
    print(f"Time taken: {time_taken} seconds")

    # Use the appropriate evaluate_paths method based on dataset type
    datasets_list = [start_dataset] + [bridge_dataset] * num_bridges + [end_dataset]
    if args.dataset == "multimodal":
        valid_paths, _ = MultiModalDataset.evaluate_paths(
            datasets_list,
            samples.cpu().numpy(),
        )
    else:  # uniform
        valid_paths, _ = UniformDataset.evaluate_paths(
            datasets_list,
            samples.cpu().numpy(),
        )
    print(f"Success Rate: {np.mean(valid_paths)}")
    # print(f"Step validities: {np.mean(step_validities)}")

    # Use the appropriate plotting method based on dataset type
    if args.dataset == "multimodal":
        fig = MultiModalDataset.plot_multi_step_transitions(
            datasets_list,
            samples.cpu().numpy(),
            annotate_valid=True,
        )
    else:  # uniform
        fig = UniformDataset.plot_multi_step_transitions(
            datasets_list,
            samples.cpu().numpy(),
            annotate_valid=True,
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_directory, exist_ok=True)
    fig.savefig(
        os.path.join(
            args.output_directory,
            f"{timestamp}.png",
        )
    )
    with open(
        os.path.join(
            args.output_directory,
            f"{timestamp}.json",
        ),
        "w",
    ) as f:
        json.dump(
            {
                "time": time_taken,
                "dataset": args.dataset,
                "horizon_length": args.horizon_length,
                "success_rate": np.mean(valid_paths),
                "samples_shape": samples.shape,
                "enable_pruning": args.enable_pruning,
                "pruning_start": args.pruning_start,
                "pruning_end": args.pruning_end,
                "pruning_top_K": args.pruning_top_K,
                "num_resampling_steps": args.num_resampling_steps,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["multimodal", "uniform"],
        default="multimodal",
        help="Dataset type: 'multimodal' for Gaussian distributions, 'uniform' for uniform distributions"
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--horizon-length", type=int, default=10)
    parser.add_argument("--num-samples-to-generate", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=100)

    parser.add_argument("--pruning-start", type=float, default=0.0)
    parser.add_argument("--enable-pruning", type=bool, default=False)
    parser.add_argument("--pruning-end", type=float, default=1.0)
    parser.add_argument("--pruning-top-K", type=float, default=0.2)
    parser.add_argument("--num-resampling-steps", type=int, default=10)

    parser.add_argument("--output-directory", type=str, default="./profile")
    args = parser.parse_args()
    main(args)
