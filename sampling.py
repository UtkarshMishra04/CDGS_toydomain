import argparse
import torch
import numpy as np
from data import MultiModalDataset, UniformDataset
from models import SimpleDiffusionModel
from samplers import CDGS
import time
import os
import json
from utils import seed_everything

# Configure device between CUDA, MPS, and CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

seed_everything(42)


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
        start_dataset, bridge_dataset, end_dataset = build_multimodal_datasets(
            num_samples, seed
        )
    else:  # uniform
        start_dataset, bridge_dataset, end_dataset = build_uniform_datasets(
            num_samples, seed
        )

    checkpoints_dir = "checkpoints_frozen"
    # Build model paths based on dataset type and training mode
    if args.training_mode == "separate":
        # Separate models for start, bridge, end
        diffusion_model_paths = {
            "start": f"./{checkpoints_dir}/{args.dataset}_start_model.pth",
            "bridge": f"./{checkpoints_dir}/{args.dataset}_bridge_model.pth",
            "end": f"./{checkpoints_dir}/{args.dataset}_end_model.pth",
        }
    else:  # unified
        # Single unified model for all components
        diffusion_model_paths = {
            "start": f"./{checkpoints_dir}/{args.dataset}_diffusion_model.pth",
            "bridge": f"./{checkpoints_dir}/{args.dataset}_diffusion_model.pth",
            "end": f"./{checkpoints_dir}/{args.dataset}_diffusion_model.pth",
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

    # Create filename based on parameters for easy identification and rerunning
    pruning_str = (
        "noprune"
        if not args.enable_pruning
        else f"prune_k{args.pruning_top_K}_s{args.pruning_start}_e{args.pruning_end}"
    )
    filename = (
        f"{args.dataset}_{args.training_mode}_"
        f"h{args.horizon_length}_"
        f"n{args.num_samples_to_generate}_"
        f"inf{args.num_inference_steps}_"
        f"res{args.num_resampling_steps}_"
        f"{pruning_str}"
    )

    os.makedirs(args.output_directory, exist_ok=True)
    fig.savefig(os.path.join(args.output_directory, f"{filename}.png"))

    with open(os.path.join(args.output_directory, f"{filename}.json"), "w") as f:
        json.dump(
            {
                "time": time_taken,
                "dataset": args.dataset,
                "training_mode": args.training_mode,
                "horizon_length": args.horizon_length,
                "num_samples_to_generate": args.num_samples_to_generate,
                "batch_size": args.batch_size,
                "success_rate": np.mean(valid_paths),
                "samples_shape": list(samples.shape),
                "enable_pruning": args.enable_pruning,
                "pruning_start": args.pruning_start,
                "pruning_end": args.pruning_end,
                "pruning_top_K": args.pruning_top_K,
                "num_resampling_steps": args.num_resampling_steps,
                "num_inference_steps": args.num_inference_steps,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified CDGS sampling script")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["multimodal", "uniform"],
        default="multimodal",
        help="Dataset type: 'multimodal' for Gaussian distributions, 'uniform' for uniform distributions",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["separate", "unified"],
        default="separate",
        help="Training mode: 'separate' for individually trained models, 'unified' for single model",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--horizon-length", type=int, default=10)
    parser.add_argument("--num-samples-to-generate", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=100)

    parser.add_argument(
        "--disable-pruning",
        action="store_true",
        help="Disable pruning (pruning is enabled by default)",
    )
    parser.add_argument("--pruning-start", type=float, default=0.0)
    parser.add_argument("--pruning-end", type=float, default=1.0)
    parser.add_argument("--pruning-top-K", type=float, default=0.2)
    parser.add_argument("--num-resampling-steps", type=int, default=10)

    parser.add_argument("--output-directory", type=str, default="./profile")
    args = parser.parse_args()

    # Convert disable_pruning to enable_pruning for backward compatibility
    args.enable_pruning = not args.disable_pruning

    main(args)
