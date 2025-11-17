import torch
from torch.utils.data import DataLoader

import numpy as np
import os
import argparse
from tqdm import tqdm
from data import MultiModalDataset, UniformDataset

# Diffusers imports: prefer scheduler-specific import paths to satisfy static checkers
# DDPMScheduler is provided under diffusers.schedulers.scheduling_ddpm in some versions.
try:
    from diffusers import DDPMScheduler
except Exception:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from models import SimpleDiffusionModel
from utils import seed_everything


def build_multimodal_start(num_samples: int, seed: int) -> MultiModalDataset:
    """Build the start dataset with Gaussian distributions."""
    return MultiModalDataset(
        num_samples=num_samples,
        start_means=[0.0],
        start_stds=[0.2],
        end_means=[0.75, -0.75],
        end_stds=[0.1, 0.1],
        transition_matrix=np.array([[1, 1]], dtype=bool),
        seed=seed,
    )


def build_multimodal_bridge(num_samples: int, seed: int) -> MultiModalDataset:
    """Build the bridge dataset with Gaussian distributions."""
    return MultiModalDataset(
        num_samples=num_samples,
        start_means=[0.75, -0.75],
        start_stds=[0.1, 0.1],
        end_means=[0.75, -0.75],
        end_stds=[0.1, 0.1],
        transition_matrix=np.array([[1, 0], [0, 1]], dtype=bool),
        seed=seed,
    )


def build_multimodal_end(num_samples: int, seed: int) -> MultiModalDataset:
    """Build the end dataset with Gaussian distributions."""
    return MultiModalDataset(
        num_samples=num_samples,
        start_means=[0.75, -0.75],
        start_stds=[0.1, 0.1],
        end_means=[0.0],
        end_stds=[0.2],
        transition_matrix=np.array([[1], [1]], dtype=bool),
        seed=seed,
    )


def build_uniform_start(num_samples: int, seed: int) -> UniformDataset:
    """Build the start dataset with uniform distributions.

    X_mid -> (X_up, X_down)
    """
    return UniformDataset(
        num_samples=num_samples,
        start_ranges=[(-0.25, 0.25)],  # X_mid
        end_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        transition_matrix=np.array([[1, 1]], dtype=bool),
        seed=seed,
    )


def build_uniform_bridge(num_samples: int, seed: int) -> UniformDataset:
    """Build the bridge dataset with uniform distributions.

    (X_up -> X_up) and (X_down -> X_down)
    """
    return UniformDataset(
        num_samples=num_samples,
        start_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        end_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        transition_matrix=np.array([[1, 0], [0, 1]], dtype=bool),
        seed=seed,
    )


def build_uniform_end(num_samples: int, seed: int) -> UniformDataset:
    """Build the end dataset with uniform distributions.

    (X_up, X_down) -> X_mid
    """
    return UniformDataset(
        num_samples=num_samples,
        start_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        end_ranges=[(-0.25, 0.25)],  # X_mid
        transition_matrix=np.array([[1], [1]], dtype=bool),
        seed=seed,
    )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a diffusion model on a single dataset component"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["multimodal", "uniform"],
        default="multimodal",
        help="Dataset type: 'multimodal' for Gaussian distributions, 'uniform' for uniform distributions",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["start", "bridge", "end"],
        required=True,
        help="Dataset component to train on: 'start', 'bridge', or 'end'",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples in the dataset"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()

    # Configure device between CUDA, MPS, and CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Set seed for reproducibility
    seed_everything(args.seed)

    # Build dataset based on arguments
    print(f"Building {args.dataset} {args.component} dataset...")

    # Map dataset type and component to builder function
    dataset_builders = {
        ("multimodal", "start"): build_multimodal_start,
        ("multimodal", "bridge"): build_multimodal_bridge,
        ("multimodal", "end"): build_multimodal_end,
        ("uniform", "start"): build_uniform_start,
        ("uniform", "bridge"): build_uniform_bridge,
        ("uniform", "end"): build_uniform_end,
    }

    builder = dataset_builders[(args.dataset, args.component)]

    # For bridge dataset, use more samples (as in original train.py)
    dataset = builder(args.num_samples, args.seed)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model and optimizer
    model = SimpleDiffusionModel().to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("training", exist_ok=True)

    print(f"Starting training loop on device: {device}")
    print(f"Dataset type: {args.dataset}")
    print(f"Component: {args.component}")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")

    accuracy = 0.0
    pbar = tqdm(range(args.num_epochs), desc="Loss: 0.0")
    for epoch in pbar:
        for batch in tqdm(dataloader, total=len(dataloader), leave=False):
            optimizer.zero_grad()
            batch = batch.to(device)
            loss = model.training_step(batch, scheduler)
            loss.backward()
            optimizer.step()

        # Save model
        if (epoch % 50 == 0 and epoch > 0) or epoch == args.num_epochs - 1:
            checkpoint_name = (
                f"{args.checkpoint_dir}/{args.dataset}_{args.component}_model.pth"
            )
            torch.save(model.state_dict(), checkpoint_name)

            # Sampling
            samples = model.sample(
                batch_size=1000,
                noise_scheduler=scheduler,
                device=device,
                num_inference_steps=50,
            )

            # Compute accuracy if available
            if hasattr(dataset, "compute_accuracy"):
                accuracy = dataset.compute_accuracy(samples)

        pbar.set_description(f"Loss: {loss.item():.4f}, Acc: {accuracy:.3f}")

    checkpoint_name = f"{args.checkpoint_dir}/{args.dataset}_{args.component}_model.pth"
    print(f"Training complete! Model saved to {checkpoint_name}")


if __name__ == "__main__":
    main()
