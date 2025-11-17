import torch
from torch.utils.data import DataLoader

import numpy as np
import os
import argparse
from tqdm import tqdm
from data import MultiModalDataset, UniformDataset
from torch.utils.data import ConcatDataset

# Diffusers imports: prefer scheduler-specific import paths to satisfy static checkers
# DDPMScheduler is provided under diffusers.schedulers.scheduling_ddpm in some versions.
try:
    from diffusers import DDPMScheduler
except Exception:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from models import SimpleDiffusionModel
from utils import seed_everything


def is_valid_transition(sample, dataset):
    """Check if a 2D sample (x1, x2) is a valid transition for the dataset.

    Args:
        sample: A numpy array of shape (2,) representing (x1, x2)
        dataset: Either a MultiModalDataset or UniformDataset

    Returns:
        bool: True if the sample is a valid transition according to the dataset
    """
    x1, x2 = sample

    if isinstance(dataset, MultiModalDataset):
        # Check all possible mode transitions
        for start_mode, end_mode in zip(*np.where(dataset.transition_matrix)):
            # Check if x1 is plausible from start_mode (using 3-sigma rule)
            start_mean = dataset.start_means[start_mode]
            start_std = dataset.start_stds[start_mode]
            start_valid = abs(x1 - start_mean) <= 3 * start_std

            # Check if x2 is plausible from end_mode
            end_mean = dataset.end_means[end_mode]
            end_std = dataset.end_stds[end_mode]
            end_valid = abs(x2 - end_mean) <= 3 * end_std

            if start_valid and end_valid:
                return True
        return False

    elif isinstance(dataset, UniformDataset):
        # Check all possible mode transitions
        for start_mode, end_mode in zip(*np.where(dataset.transition_matrix)):
            # Check if x1 is in start range
            start_low, start_high = dataset.start_ranges[start_mode]
            start_valid = start_low <= x1 <= start_high

            # Check if x2 is in end range
            end_low, end_high = dataset.end_ranges[end_mode]
            end_valid = end_low <= x2 <= end_high

            if start_valid and end_valid:
                return True
        return False

    return False


def compute_accuracy_any(samples, datasets):
    """Compute accuracy by checking if each sample is valid under ANY dataset.

    Args:
        samples: Tensor of shape (N, 2) containing N samples
        datasets: List of datasets to check against

    Returns:
        float: Fraction of samples that are valid under at least one dataset
    """
    samples_np = samples.cpu().numpy()
    valid_count = 0

    for sample in samples_np:
        # Check if sample is valid under ANY of the component datasets
        if any(is_valid_transition(sample, ds) for ds in datasets):
            valid_count += 1

    return valid_count / len(samples_np)


def build_multimodal_datasets(num_samples: int, seed: int) -> ConcatDataset:
    """Build the original Gaussian multimodal datasets.

    Args:
        num_samples: Number of samples per dataset
        seed: Random seed

    Returns:
        ConcatDataset containing start, bridge, and end datasets
    """
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
        num_samples=num_samples * 4,
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

    datasets = [start_dataset, bridge_dataset, end_dataset]
    return ConcatDataset(datasets)


def build_uniform_datasets(num_samples: int, seed: int) -> ConcatDataset:
    """Build the new uniform distribution datasets.

    Args:
        num_samples: Number of samples per dataset
        seed: Random seed

    Returns:
        ConcatDataset containing start, bridge, and end datasets

    Dataset structure:
        - X_up: 0.5 to 1.0
        - X_mid: -0.25 to 0.25
        - X_down: -1.0 to -0.5
    """
    start_dataset = UniformDataset(
        num_samples=num_samples,
        start_ranges=[(-0.25, 0.25)],  # X_mid
        end_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        transition_matrix=np.array([[1, 1]], dtype=bool),  # X_mid can go to X_up or X_down
        seed=seed,
    )

    bridge_dataset = UniformDataset(
        num_samples=num_samples * 4,
        start_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        end_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        transition_matrix=np.array([[1, 0], [0, 1]], dtype=bool),  # X_up->X_up, X_down->X_down
        seed=seed,
    )

    end_dataset = UniformDataset(
        num_samples=num_samples,
        start_ranges=[(0.5, 1.0), (-1.0, -0.5)],  # X_up, X_down
        end_ranges=[(-0.25, 0.25)],  # X_mid
        transition_matrix=np.array([[1], [1]], dtype=bool),  # Both can go to X_mid
        seed=seed,
    )

    datasets = [start_dataset, bridge_dataset, end_dataset]
    return ConcatDataset(datasets)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a diffusion model on synthetic datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["multimodal", "uniform"],
        default="multimodal",
        help="Dataset type: 'multimodal' for Gaussian distributions, 'uniform' for uniform distributions"
    )
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
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

    # Build dataset based on argument
    print(f"Building {args.dataset} datasets...")
    if args.dataset == "multimodal":
        concat_dataset = build_multimodal_datasets(args.num_samples, args.seed)
    else:  # uniform
        concat_dataset = build_uniform_datasets(args.num_samples, args.seed)

    # Extract component datasets for accuracy computation
    component_datasets = list(concat_dataset.datasets)

    dataloader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model and optimizer
    model = SimpleDiffusionModel().to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("training", exist_ok=True)

    print(f"Starting training loop on device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")

    accuracy = 0.0
    best_loss = float('inf')
    pbar = tqdm(range(args.num_epochs), desc="Loss: 0.0")
    for epoch in pbar:
        epoch_loss = 0.0
        num_batches = 0
        for batch in tqdm(dataloader, total=len(dataloader), leave=False):
            optimizer.zero_grad()
            batch = batch.to(device)
            loss = model.training_step(batch, scheduler)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches

        # Save best model if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_name = f"checkpoints/{args.dataset}_diffusion_model_best.pth"
            torch.save(model.state_dict(), best_checkpoint_name)

        # Save model periodically
        if (epoch % 50 == 0 and epoch > 0) or epoch == args.num_epochs - 1:
            checkpoint_name = f"checkpoints/{args.dataset}_diffusion_model.pth"
            torch.save(model.state_dict(), checkpoint_name)

            # Sampling and accuracy computation
            samples = model.sample(
                batch_size=1000,
                noise_scheduler=scheduler,
                device=device,
                num_inference_steps=50,
            )
            # Compute accuracy: sample is valid if it's valid under ANY component dataset
            accuracy = compute_accuracy_any(samples, component_datasets)

        pbar.set_description(f"Loss: {avg_loss:.4f}, Best: {best_loss:.4f}, Acc: {accuracy:.3f}")

    print(f"Training complete! Model saved to checkpoints/{args.dataset}_diffusion_model.pth")


if __name__ == "__main__":
    main()
